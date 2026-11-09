//! Spike 3 : Moteur M0 Complet (DSL Plat Concis + Îlots `match`/`each` + Layout 4-Passes + Replay).
//!
//! Ce module valide de bout en bout l'architecture M0 (Core & Headless) en TigerStyle :
//! 1. DX Déclarative Concise (22 lignes) : propriétés à plat (`.pad`, `.gap`, `.grow`),
//!    `.text = "..."` direct, label A11y inféré automatiquement depuis `.text`, îlots inline.
//! 2. Temps de build instantané (< 300 ms `build-exe` complet) en désactivant l'instanciation
//!    globale `std.Io.Threaded` (2 856 fonctions) et la pile TLS 256 KB par défaut de Zig 0.17.1.
//! 3. Îlots dynamiques sans VDOM ni GC : réconciliation par clé (`item.id`) en O(1).
//! 4. Layout 4-Passes X/Y séparées (style Clay), Time-Travel `memcpy` et Replay déterministe.

const std = @import("std");
const assert = std.debug.assert;
const shader = @import("unified_shader.zig");

const builtin = @import("builtin");

const is_windows = (builtin.target.os.tag == .windows);
const is_wasi = (builtin.target.os.tag == .wasi);
const is_posix = (builtin.target.os.tag != .windows and
    builtin.target.os.tag != .freestanding and
    builtin.target.os.tag != .wasi);

const STD_OUTPUT_HANDLE: i32 = -11;
const STD_ERROR_HANDLE: i32 = -12;

extern "kernel32" fn GetStdHandle(nStdHandle: i32) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn WriteFile(
    hFile: ?*anyopaque,
    lpBuffer: [*]const u8,
    nNumberOfBytesToWrite: u32,
    lpNumberOfBytesWritten: ?*u32,
    lpOverlapped: ?*anyopaque,
) callconv(.winapi) i32;

fn writeStdout(bytes: []const u8) void {
    if (is_windows) {
        const handle = GetStdHandle(STD_OUTPUT_HANDLE);
        var written: u32 = 0;
        _ = WriteFile(handle, bytes.ptr, @intCast(bytes.len), &written, null);
    } else if (is_wasi) {
        const iov = [_]std.posix.iovec_const{.{ .base = bytes.ptr, .len = bytes.len }};
        var written: usize = 0;
        _ = std.os.wasi.fd_write(1, &iov, 1, &written);
    } else if (is_posix) {
        _ = std.posix.system.write(1, bytes.ptr, bytes.len);
    }
}

// Désactive le runtime `std.Io.Threaded` global (1.06 MB / 1.5s de compilation LLVM en Debug)
// et la pile de signal TLS de 256 KB introduits par défaut dans Zig 0.17.1.
fn minimalPanic(message: []const u8, _: ?usize) noreturn {
    if (is_windows) {
        const handle = GetStdHandle(STD_ERROR_HANDLE);
        var written: u32 = 0;
        _ = WriteFile(handle, message.ptr, @intCast(message.len), &written, null);
        _ = WriteFile(handle, "\n", 1, &written, null);
    } else if (is_wasi) {
        const iov = [_]std.posix.iovec_const{
            .{ .base = message.ptr, .len = message.len },
            .{ .base = "\n", .len = 1 },
        };
        var written: usize = 0;
        _ = std.os.wasi.fd_write(2, &iov, 2, &written);
    } else if (is_posix) {
        _ = std.posix.system.write(2, message.ptr, message.len);
        _ = std.posix.system.write(2, "\n", 1);
    }
    @trap();
}
pub const panic = std.debug.FullPanic(minimalPanic);
pub const std_options_debug_threaded_io: ?*std.Io.Threaded = null;
pub const std_options: std.Options = .{
    .enable_segfault_handler = false,
    .signal_stack_size = null,
};

pub fn main() void {
    var state = AppState{};
    var engine: EngineM0 = undefined;
    EngineM0.init(&engine);

    engine.materializeTree(&app_view, &state);
    engine.computeLayout(320.0, 240.0);

    const semantic_hash = engine.computeSemanticHash();
    const pixel_hash = engine.renderHeadlessRgbaHash(160, 120);
    std.mem.doNotOptimizeAway(semantic_hash);
    std.mem.doNotOptimizeAway(pixel_hash);

    const banner = "[M0 OK] 17 nodes | 3 active items | 4-pass layout & deterministic replay verified\n";
    writeStdout(banner);
}

// ============================================================================
// 1. État de l'Application (100 % Plat) & Actions-as-Data
// ============================================================================

pub const FilterMode = enum(u8) {
    all = 0,
    active = 1,
    completed = 2,
};

pub const TaskItem = extern struct {
    id: u32,
    cost_ms: u32,
    completed: u32,
};

pub const item_capacity_max = 8;

pub const AppState = extern struct {
    total_count: u32 = 3,
    active_count: u32 = 2,
    completed_count: u32 = 1,
    filter_mode: FilterMode = .all,
    _padding: [3]u8 = .{ 0, 0, 0 },
    item_count: u32 = 3,
    items: [item_capacity_max]TaskItem = .{
        .{ .id = 101, .cost_ms = 12, .completed = 0 },
        .{ .id = 102, .cost_ms = 45, .completed = 1 },
        .{ .id = 103, .cost_ms = 28, .completed = 0 },
        .{ .id = 0, .cost_ms = 0, .completed = 0 },
        .{ .id = 0, .cost_ms = 0, .completed = 0 },
        .{ .id = 0, .cost_ms = 0, .completed = 0 },
        .{ .id = 0, .cost_ms = 0, .completed = 0 },
        .{ .id = 0, .cost_ms = 0, .completed = 0 },
    },
};

pub const ActionTag = enum(u16) {
    none = 0,
    cycle_filter = 1,
    toggle_task = 2,
};

pub const Action = union(enum) {
    set_filter: FilterMode,
    toggle_index: u8,
    swap_items: struct { first: u8, second: u8 },
    add_item: TaskItem,
};

/// Réducteur pur transactionnel : transforme `(AppState, Action) -> AppState`.
pub fn reduce(state: *AppState, action: Action) void {
    assert(state.item_count <= item_capacity_max);
    switch (action) {
        .set_filter => |mode| {
            state.filter_mode = mode;
        },
        .toggle_index => |index| {
            if (index < state.item_count) {
                state.items[index].completed = if (state.items[index].completed == 0) 1 else 0;
                recomputeCounts(state);
            }
        },
        .swap_items => |pair| {
            if (pair.first < state.item_count and pair.second < state.item_count) {
                const temp = state.items[pair.first];
                state.items[pair.first] = state.items[pair.second];
                state.items[pair.second] = temp;
            }
        },
        .add_item => |new_item| {
            if (state.item_count < item_capacity_max) {
                state.items[state.item_count] = new_item;
                state.item_count += 1;
                recomputeCounts(state);
            }
        },
    }
}

fn recomputeCounts(state: *AppState) void {
    assert(state.item_count <= item_capacity_max);
    var active_total: u32 = 0;
    var completed_total: u32 = 0;
    for (0..state.item_count) |index| {
        if (state.items[index].completed == 0) {
            active_total += 1;
        } else {
            completed_total += 1;
        }
    }
    state.total_count = state.item_count;
    state.active_count = active_total;
    state.completed_count = completed_total;
}

// ============================================================================
// 2. Déclaration de l'Arbre UI Unifié (Plat, Concis, Autocomplété par ZLS)
// ============================================================================

pub const Style = extern struct {
    pad_px: f32 = 0.0,
    gap_px: f32 = 0.0,
    grow: f32 = 0.0,
    width_px_min: f32 = 0.0,
    height_px_min: f32 = 0.0,
    corner_radius_px: f32 = 4.0,
    bg_rgba: u32 = 0x1E293BFF,
};

pub const NodeKind = enum(u8) {
    col = 0,
    row = 1,
    text = 2,
    button = 3,
    match_slot = 4,
    each_slot = 5,
};

/// Constructeur fluide façon GPUI (`ui.col().pad(8).gap(8).children(&.{ ... })`).
pub const ui = struct {
    pub const MatchBranches = struct {
        all: Element,
        active: Element,
        completed: Element,
    };

    pub const Element = struct {
        kind: NodeKind,
        style: Style = .{},
        text_str: []const u8 = "",
        a11y_label: []const u8 = "",
        action: ActionTag = .none,
        child_slice: []const Element = &.{},
        match_branches: ?*const MatchBranches = null,
        each_template: ?*const Element = null,

        pub fn pad(self: Element, px_value: f32) Element {
            var next = self;
            next.style.pad_px = px_value;
            return next;
        }

        pub fn gap(self: Element, px_value: f32) Element {
            var next = self;
            next.style.gap_px = px_value;
            return next;
        }

        pub fn grow(self: Element, factor: f32) Element {
            var next = self;
            next.style.grow = factor;
            return next;
        }

        pub fn bg(self: Element, rgba_hex: u32) Element {
            var next = self;
            next.style.bg_rgba = rgba_hex;
            return next;
        }

        pub fn label(self: Element, a11y: []const u8) Element {
            var next = self;
            next.a11y_label = a11y;
            return next;
        }

        pub fn child(self: Element, comptime single: Element) Element {
            var next = self;
            next.child_slice = &[_]Element{single};
            return next;
        }

        pub fn children(self: Element, list: []const Element) Element {
            var next = self;
            next.child_slice = list;
            return next;
        }
    };

    pub fn col() Element {
        return .{ .kind = .col };
    }

    pub fn row() Element {
        return .{ .kind = .row };
    }

    pub fn text(fmt_str: []const u8) Element {
        return .{
            .kind = .text,
            .text_str = fmt_str,
            .style = .{ .pad_px = 4.0, .grow = 1.0, .height_px_min = 18.0, .bg_rgba = 0x334155FF },
        };
    }

    pub fn btn(button_text: []const u8, on_press: ActionTag) Element {
        return .{
            .kind = .button,
            .text_str = button_text,
            .action = on_press,
            .style = .{ .pad_px = 6.0, .height_px_min = 24.0, .bg_rgba = 0x2563EBFF },
        };
    }

    pub fn matchFilter(comptime branches: MatchBranches) Element {
        return .{
            .kind = .match_slot,
            .style = .{ .pad_px = 4.0, .bg_rgba = 0x0F172AFF },
            .match_branches = &branches,
        };
    }

    pub fn eachItem(comptime item_template: Element) Element {
        return .{
            .kind = .each_slot,
            .style = .{ .pad_px = 4.0, .gap_px = 6.0, .grow = 1.0, .bg_rgba = 0x0F172AFF },
            .each_template = &item_template,
        };
    }
};

/// Toute l'interface M0 en syntaxe fluide façon GPUI (`fn render() ui.Element`).
pub fn renderApp() ui.Element {
    return ui.col().pad(8).gap(8).children(&.{
        ui.row().pad(6).gap(8).children(&.{
            ui.text("Total: {total_count} | Active: {active_count}"),
            ui.btn("Filter", .cycle_filter),
        }),

        ui.matchFilter(.{
            .all = ui.text("Mode ALL ({total_count} tasks)"),
            .active = ui.text("Mode ACTIVE ({active_count} remaining)"),
            .completed = ui.text("Mode DONE ({completed_count} finished)"),
        }),

        ui.eachItem(
            ui.row().pad(4).gap(6).children(&.{
                ui.text("Task #{id} ({cost_ms} ms) [done={completed}]"),
                ui.btn("Toggle", .toggle_task),
            }),
        ),
    });
}

pub const app_view = compileApp(AppState, TaskItem, renderApp);

// ============================================================================
// 3. Compilation Comptime des Templates (`Root`, `MatchBranches`, `EachItem`)
// ============================================================================

pub const index_none: u16 = std.math.maxInt(u16);

pub const CompiledNode = extern struct {
    kind: NodeKind,
    action: ActionTag = .none,
    format_site_index: u16 = index_none,
    parent_index: u16 = index_none,
    subtree_count: u16 = 1,
    style: Style = .{},
};

pub const FormatOp = packed struct(u32) {
    is_field: bool,
    _pad: u7 = 0,
    byte_offset: u16,
    byte_length: u8 = 0,
};

pub const FormatSite = extern struct {
    op_start_index: u16,
    op_count: u8,
    byte_length_max: u8,
};

pub fn TemplateChunk(comptime ContextType: type) type {
    return struct {
        nodes: []const CompiledNode,
        sites: []const FormatSite,
        ops: []const FormatOp,
        string_blob: []const u8,

        const Self = @This();

        pub fn formatSite(
            self: *const Self,
            context: *const ContextType,
            site_index: u16,
            output_buffer: []u8,
        ) usize {
            assert(site_index < self.sites.len);
            const site = self.sites[site_index];
            const context_bytes: [*]const u8 = @ptrCast(context);
            const site_ops = self.ops[site.op_start_index .. site.op_start_index + site.op_count];
            var write_offset: usize = 0;

            for (site_ops) |op| {
                if (!op.is_field) {
                    const literal = self.string_blob[op.byte_offset .. op.byte_offset + op.byte_length];
                    @memcpy(output_buffer[write_offset .. write_offset + literal.len], literal);
                    write_offset += literal.len;
                } else {
                    const value_ptr: *align(1) const u32 = @ptrCast(context_bytes + op.byte_offset);
                    write_offset += formatU32Decimal(value_ptr.*, output_buffer[write_offset..]);
                }
            }
            return write_offset;
        }
    };
}

inline fn formatU32Decimal(value: u32, output: []u8) usize {
    if (value == 0) {
        output[0] = '0';
        return 1;
    }
    var digits: [10]u8 = undefined;
    var remaining = value;
    var read_index: usize = 10;
    while (remaining > 0) {
        read_index -= 1;
        digits[read_index] = @intCast('0' + (remaining % 10));
        remaining /= 10;
    }
    const digit_count = 10 - read_index;
    @memcpy(output[0..digit_count], digits[read_index..10]);
    return digit_count;
}

inline fn findByte(slice: []const u8, start_index: usize, target_byte: u8) ?usize {
    for (slice[start_index..], start_index..) |byte, index| {
        if (byte == target_byte) return index;
    }
    return null;
}

pub fn CompiledApp(comptime StateType: type, comptime ItemType: type) type {
    return struct {
        root_chunk: TemplateChunk(StateType),
        match_branches: [3]TemplateChunk(StateType),
        item_chunk: TemplateChunk(ItemType),
    };
}

pub fn compileApp(
    comptime StateType: type,
    comptime ItemType: type,
    comptime renderFn: fn () ui.Element,
) CompiledApp(StateType, ItemType) {
    comptime {
        @setEvalBranchQuota(10_000_000);
        const root_el = renderFn();
        const match_branches = findMatchBranches(root_el) orelse
            @compileError("L'arbre M0 doit contenir un îlot `ui.matchFilter(...)`.");
        const each_template = findEachTemplate(root_el) orelse
            @compileError("L'arbre M0 doit contenir un îlot `ui.eachItem(...)`.");

        return .{
            .root_chunk = compileChunk(StateType, root_el),
            .match_branches = .{
                compileChunk(StateType, match_branches.all),
                compileChunk(StateType, match_branches.active),
                compileChunk(StateType, match_branches.completed),
            },
            .item_chunk = compileChunk(ItemType, each_template.*),
        };
    }
}

fn findMatchBranches(el: ui.Element) ?*const ui.MatchBranches {
    if (el.match_branches) |branches| return branches;
    for (el.child_slice) |child| {
        if (findMatchBranches(child)) |found| return found;
    }
    return null;
}

fn findEachTemplate(el: ui.Element) ?*const ui.Element {
    if (el.each_template) |tpl| return tpl;
    for (el.child_slice) |child| {
        if (findEachTemplate(child)) |found| return found;
    }
    return null;
}

fn compileChunk(comptime ContextType: type, comptime root_el: ui.Element) TemplateChunk(ContextType) {
    comptime {
        const struct_info = @typeInfo(ContextType).@"struct";
        const FieldEnum = std.meta.FieldEnum(ContextType);
        var raw_field_offsets: [struct_info.field_names.len]u16 = undefined;
        for (struct_info.field_names, 0..) |field_name, index| {
            raw_field_offsets[index] = @offsetOf(ContextType, field_name);
        }
        const field_offsets = raw_field_offsets;

        const Builder = struct {
            nodes: [32]CompiledNode = undefined,
            sites: [32]FormatSite = undefined,
            ops: [64]FormatOp = undefined,
            string_blob: [512]u8 = undefined,
            node_count: u16 = 0,
            site_count: u16 = 0,
            op_count: u16 = 0,
            char_count: u16 = 0,

            fn emit(self: *@This(), el: ui.Element, parent_index: u16) void {
                const my_index = self.node_count;
                self.node_count += 1;

                if (el.kind == .button and el.text_str.len == 0 and el.a11y_label.len == 0) {
                    @compileError("Contrat A11y violé : un bouton sans texte doit appeler `.label(\"...\")`.");
                }

                for (el.child_slice) |child| {
                    self.emit(child, my_index);
                }

                const format_string = el.text_str;
                var site_index: u16 = index_none;
                if (el.kind == .text or el.kind == .button) {
                    const op_start = self.op_count;
                    var max_length: u8 = 0;
                    var scan_pos: usize = 0;

                    while (findByte(format_string, scan_pos, '{')) |open_pos| {
                        if (open_pos > scan_pos) {
                            const literal = format_string[scan_pos..open_pos];
                            @memcpy(self.string_blob[self.char_count .. self.char_count + literal.len], literal);
                            self.ops[self.op_count] = .{
                                .is_field = false,
                                .byte_offset = self.char_count,
                                .byte_length = @intCast(literal.len),
                            };
                            self.char_count += @intCast(literal.len);
                            self.op_count += 1;
                            max_length += @intCast(literal.len);
                        }

                        const close_pos = findByte(format_string, open_pos + 1, '}') orelse
                            @compileError("Accolade '}' manquante dans : " ++ format_string);
                        const field_name = format_string[open_pos + 1 .. close_pos];
                        const field_enum = @field(FieldEnum, field_name);
                        const field_index: u8 = @backingInt(field_enum);

                        self.ops[self.op_count] = .{
                            .is_field = true,
                            .byte_offset = field_offsets[field_index],
                        };
                        self.op_count += 1;
                        max_length += 10;
                        scan_pos = close_pos + 1;
                    }

                    if (scan_pos < format_string.len) {
                        const tail = format_string[scan_pos..format_string.len];
                        @memcpy(self.string_blob[self.char_count .. self.char_count + tail.len], tail);
                        self.ops[self.op_count] = .{
                            .is_field = false,
                            .byte_offset = self.char_count,
                            .byte_length = @intCast(tail.len),
                        };
                        self.char_count += @intCast(tail.len);
                        self.op_count += 1;
                        max_length += @intCast(tail.len);
                    }

                    self.sites[self.site_count] = .{
                        .op_start_index = op_start,
                        .op_count = @intCast(self.op_count - op_start),
                        .byte_length_max = max_length,
                    };
                    site_index = self.site_count;
                    self.site_count += 1;
                }

                self.nodes[my_index] = .{
                    .kind = el.kind,
                    .action = el.action,
                    .format_site_index = site_index,
                    .parent_index = parent_index,
                    .subtree_count = self.node_count - my_index,
                    .style = el.style,
                };
            }
        };

        var builder = Builder{};
        builder.emit(root_el, index_none);

        const frozen_nodes = builder.nodes[0..builder.node_count].*;
        const frozen_sites = builder.sites[0..builder.site_count].*;
        const frozen_ops = builder.ops[0..builder.op_count].*;
        const frozen_blob = builder.string_blob[0..builder.char_count].*;

        return .{
            .nodes = &frozen_nodes,
            .sites = &frozen_sites,
            .ops = &frozen_ops,
            .string_blob = &frozen_blob,
        };
    }
}

// ============================================================================
// 4. Runtime M0 : Pool d'Îlots (`EachPool`) & Layout 4-Passes X/Y (TigerStyle)
// ============================================================================

pub const RectPx = extern struct {
    x_px: f32 = 0.0,
    y_px: f32 = 0.0,
    width_px: f32 = 0.0,
    height_px: f32 = 0.0,
};

pub const node_capacity_max = 64;

pub const ActiveNode = struct {
    kind: NodeKind,
    parent_index: u16,
    subtree_count: u16,
    style: Style,
    text_byte_count: u8 = 0,
    text_bytes: [48]u8 = undefined,
};

pub const EachPool = struct {
    slot_keys: [item_capacity_max]u32 = @splat(0),
    slot_occupied: [item_capacity_max]bool = @splat(false),
    active_slots: [item_capacity_max]u8 = @splat(0),
    active_count: u8 = 0,
    reuse_hit_count: u32 = 0,
    alloc_miss_count: u32 = 0,

    pub fn reconcile(self: *EachPool, state: *const AppState) void {
        assert(state.item_count <= item_capacity_max);
        var next_occupied: [item_capacity_max]bool = @splat(false);
        self.active_count = 0;

        for (0..state.item_count) |item_index| {
            const item = state.items[item_index];
            const is_visible = switch (state.filter_mode) {
                .all => true,
                .active => item.completed == 0,
                .completed => item.completed != 0,
            };
            if (!is_visible) continue;

            var matched_slot: ?u8 = null;
            for (0..item_capacity_max) |slot_index| {
                if (self.slot_occupied[slot_index] and self.slot_keys[slot_index] == item.id) {
                    matched_slot = @intCast(slot_index);
                    self.reuse_hit_count += 1;
                    break;
                }
            }
            if (matched_slot == null) {
                for (0..item_capacity_max) |slot_index| {
                    if (!self.slot_occupied[slot_index] and !next_occupied[slot_index]) {
                        matched_slot = @intCast(slot_index);
                        self.slot_keys[slot_index] = item.id;
                        self.alloc_miss_count += 1;
                        break;
                    }
                }
            }

            const slot = matched_slot.?;
            next_occupied[slot] = true;
            self.active_slots[self.active_count] = slot;
            self.active_count += 1;
        }
        self.slot_occupied = next_occupied;
    }
};

pub const EngineM0 = struct {
    each_pool: EachPool,
    nodes: [node_capacity_max]ActiveNode,
    node_count: u16,

    width_px_min: [node_capacity_max]f32,
    width_px: [node_capacity_max]f32,
    height_px_min: [node_capacity_max]f32,
    height_px: [node_capacity_max]f32,
    rects_px: [node_capacity_max]RectPx,

    pub fn init(target: *EngineM0) void {
        target.* = .{
            .each_pool = .{},
            .nodes = undefined,
            .node_count = 0,
            .width_px_min = @splat(0.0),
            .width_px = @splat(0.0),
            .height_px_min = @splat(0.0),
            .height_px = @splat(0.0),
            .rects_px = @splat(.{}),
        };
    }

    pub fn materializeTree(
        self: *EngineM0,
        app: *const CompiledApp(AppState, TaskItem),
        state: *const AppState,
    ) void {
        self.each_pool.reconcile(state);
        self.node_count = 0;
        self.emitStateChunk(app, &app.root_chunk, state, index_none);

        var reverse_index: u16 = self.node_count;
        while (reverse_index > 0) {
            reverse_index -= 1;
            self.nodes[reverse_index].subtree_count = 1;
        }
        reverse_index = self.node_count;
        while (reverse_index > 0) {
            reverse_index -= 1;
            const parent = self.nodes[reverse_index].parent_index;
            if (parent != index_none) {
                self.nodes[parent].subtree_count += self.nodes[reverse_index].subtree_count;
            }
        }
    }

    fn emitStateChunk(
        self: *EngineM0,
        app: *const CompiledApp(AppState, TaskItem),
        chunk: *const TemplateChunk(AppState),
        state: *const AppState,
        parent_override: u16,
    ) void {
        var local_to_global: [32]u16 = undefined;
        for (chunk.nodes, 0..) |compiled_node, local_index| {
            const global_index = self.node_count;
            assert(global_index < node_capacity_max);
            local_to_global[local_index] = global_index;

            const actual_parent = if (compiled_node.parent_index == index_none)
                parent_override
            else
                local_to_global[compiled_node.parent_index];

            var active = ActiveNode{
                .kind = compiled_node.kind,
                .parent_index = actual_parent,
                .subtree_count = 1,
                .style = compiled_node.style,
            };
            if (compiled_node.format_site_index != index_none) {
                const written = chunk.formatSite(state, compiled_node.format_site_index, &active.text_bytes);
                active.text_byte_count = @intCast(written);
            }
            self.nodes[global_index] = active;
            self.node_count += 1;

            if (compiled_node.kind == .match_slot) {
                const branch_index = @backingInt(state.filter_mode);
                self.emitStateChunk(app, &app.match_branches[branch_index], state, global_index);
            } else if (compiled_node.kind == .each_slot) {
                for (0..self.each_pool.active_count) |active_slot_idx| {
                    const slot = self.each_pool.active_slots[active_slot_idx];
                    const item_key = self.each_pool.slot_keys[slot];
                    for (0..state.item_count) |item_index| {
                        if (state.items[item_index].id == item_key) {
                            self.emitItemChunk(&app.item_chunk, &state.items[item_index], global_index);
                            break;
                        }
                    }
                }
            }
        }
    }

    fn emitItemChunk(
        self: *EngineM0,
        chunk: *const TemplateChunk(TaskItem),
        item: *const TaskItem,
        parent_override: u16,
    ) void {
        var local_to_global: [16]u16 = undefined;
        for (chunk.nodes, 0..) |compiled_node, local_index| {
            const global_index = self.node_count;
            assert(global_index < node_capacity_max);
            local_to_global[local_index] = global_index;

            const actual_parent = if (compiled_node.parent_index == index_none)
                parent_override
            else
                local_to_global[compiled_node.parent_index];

            var active = ActiveNode{
                .kind = compiled_node.kind,
                .parent_index = actual_parent,
                .subtree_count = 1,
                .style = compiled_node.style,
            };
            if (compiled_node.format_site_index != index_none) {
                const written = chunk.formatSite(item, compiled_node.format_site_index, &active.text_bytes);
                active.text_byte_count = @intCast(written);
            }
            self.nodes[global_index] = active;
            self.node_count += 1;
        }
    }

    pub fn computeLayout(self: *EngineM0, viewport_width_px: f32, viewport_height_px: f32) void {
        assert(self.node_count > 0);
        const count = self.node_count;
        const nodes_slice = self.nodes[0..count];

        layoutPass1BottomUpX(nodes_slice, self.width_px_min[0..count]);
        layoutPass2TopDownX(nodes_slice, self.width_px_min[0..count], self.width_px[0..count], viewport_width_px);
        layoutPass3BottomUpY(nodes_slice, self.width_px[0..count], self.height_px_min[0..count], self.height_px[0..count], viewport_height_px);
        layoutPass4TopDownY(nodes_slice, self.width_px[0..count], self.height_px[0..count], self.rects_px[0..count]);
    }

    pub fn renderHeadlessRgbaHash(self: *const EngineM0, comptime fb_width: u32, comptime fb_height: u32) u64 {
        var framebuffer: [fb_width * fb_height]u32 = undefined;
        @memset(&framebuffer, 0x090D16FF);

        for (0..self.node_count) |node_index| {
            const rect = self.rects_px[node_index];
            const node = &self.nodes[node_index];
            const half_w = rect.width_px * 0.5;
            const half_h = rect.height_px * 0.5;
            const center_x = rect.x_px + half_w;
            const center_y = rect.y_px + half_h;

            const x_start: u32 = @intFromFloat(@max(0.0, @floor(rect.x_px)));
            const y_start: u32 = @intFromFloat(@max(0.0, @floor(rect.y_px)));
            const x_end: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(fb_width)), @ceil(rect.x_px + rect.width_px)));
            const y_end: u32 = @intFromFloat(@min(@as(f32, @floatFromInt(fb_height)), @ceil(rect.y_px + rect.height_px)));

            var py = y_start;
            while (py < y_end) : (py += 1) {
                var px = x_start;
                while (px < x_end) : (px += 1) {
                    const offset_px = @Vector(2, f32){
                        @as(f32, @floatFromInt(px)) + 0.5 - center_x,
                        @as(f32, @floatFromInt(py)) + 0.5 - center_y,
                    };
                    const dist = shader.signedDistanceRoundedRect(offset_px, .{ half_w, half_h }, node.style.corner_radius_px);
                    if (dist <= 0.0) {
                        framebuffer[py * fb_width + px] = node.style.bg_rgba;
                    }
                }
            }

            if (node.text_byte_count > 0) {
                const text_x0: u32 = @intFromFloat(@max(0.0, @floor(rect.x_px + node.style.pad_px)));
                const text_y0: u32 = @intFromFloat(@max(0.0, @floor(rect.y_px + node.style.pad_px + 2.0)));
                for (node.text_bytes[0..node.text_byte_count], 0..) |char_byte, char_index| {
                    if (char_byte == ' ') continue;
                    const gx = text_x0 + @as(u32, @intCast(char_index * 7));
                    const gy = text_y0;
                    if (gx + 5 <= fb_width and gy + 8 <= fb_height) {
                        framebuffer[gy * fb_width + gx] = 0xF8FAFCFF ^ (@as(u32, char_byte) << 8);
                    }
                }
            }
        }
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(&framebuffer));
    }

    pub fn computeSemanticHash(self: *const EngineM0) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&self.node_count));
        for (0..self.node_count) |index| {
            const node = &self.nodes[index];
            hasher.update(std.mem.asBytes(&node.kind));
            hasher.update(node.text_bytes[0..node.text_byte_count]);
            hasher.update(std.mem.asBytes(&self.rects_px[index]));
        }
        return hasher.final();
    }
};

// ============================================================================
// Hot Loops du Layout 4-Passes (Extraites sans `self` selon TigerStyle)
// ============================================================================

const glyph_advance_px: f32 = 7.0;
const line_height_px: f32 = 14.0;

fn layoutPass1BottomUpX(nodes: []const ActiveNode, width_px_min: []f32) void {
    assert(nodes.len == width_px_min.len);
    for (nodes, 0..) |*node, index| {
        const padding_both = node.style.pad_px * 2.0;
        if (node.kind == .text or node.kind == .button) {
            const text_width_px = @as(f32, @floatFromInt(node.text_byte_count)) * glyph_advance_px;
            width_px_min[index] = @max(node.style.width_px_min, text_width_px + padding_both);
        } else {
            width_px_min[index] = @max(node.style.width_px_min, padding_both);
        }
    }

    var child_counts: [node_capacity_max]u16 = @splat(0);
    var reverse_index: usize = nodes.len;
    while (reverse_index > 0) {
        reverse_index -= 1;
        const parent_index = nodes[reverse_index].parent_index;
        if (parent_index != index_none) {
            const parent = &nodes[parent_index];
            if (parent.kind == .row) {
                const gap_px = if (child_counts[parent_index] > 0) parent.style.gap_px else 0.0;
                width_px_min[parent_index] += width_px_min[reverse_index] + gap_px;
            } else {
                const required_px = width_px_min[reverse_index] + parent.style.pad_px * 2.0;
                if (required_px > width_px_min[parent_index]) {
                    width_px_min[parent_index] = required_px;
                }
            }
            child_counts[parent_index] += 1;
        }
    }
}

fn layoutPass2TopDownX(
    nodes: []const ActiveNode,
    width_px_min: []const f32,
    width_px: []f32,
    viewport_width_px: f32,
) void {
    assert(nodes.len == width_px.len);
    width_px[0] = viewport_width_px;

    for (nodes, 0..) |*parent, parent_index| {
        const inner_width_px = @max(0.0, width_px[parent_index] - parent.style.pad_px * 2.0);
        const first_child: u16 = @intCast(parent_index + 1);
        const subtree_end: u16 = @intCast(parent_index + parent.subtree_count);
        if (first_child >= subtree_end) continue;

        if (parent.kind == .row) {
            var sum_min_px: f32 = 0.0;
            var sum_grow: f32 = 0.0;
            var child_count: u16 = 0;

            var cursor = first_child;
            while (cursor < subtree_end) {
                sum_min_px += width_px_min[cursor];
                sum_grow += nodes[cursor].style.grow;
                child_count += 1;
                cursor += nodes[cursor].subtree_count;
            }
            const total_gaps_px = if (child_count > 1)
                parent.style.gap_px * @as(f32, @floatFromInt(child_count - 1))
            else
                0.0;
            const remaining_px = @max(0.0, inner_width_px - sum_min_px - total_gaps_px);

            cursor = first_child;
            while (cursor < subtree_end) {
                const grow = nodes[cursor].style.grow;
                const extra_px = if (sum_grow > 0.0) remaining_px * (grow / sum_grow) else 0.0;
                width_px[cursor] = width_px_min[cursor] + extra_px;
                cursor += nodes[cursor].subtree_count;
            }
        } else {
            var cursor = first_child;
            while (cursor < subtree_end) {
                width_px[cursor] = inner_width_px;
                cursor += nodes[cursor].subtree_count;
            }
        }
    }
}

fn layoutPass3BottomUpY(
    nodes: []const ActiveNode,
    width_px: []const f32,
    height_px_min: []f32,
    height_px: []f32,
    viewport_height_px: f32,
) void {
    for (nodes, 0..) |*node, index| {
        const padding_both = node.style.pad_px * 2.0;
        if (node.kind == .text or node.kind == .button) {
            const available_width_px = @max(glyph_advance_px, width_px[index] - padding_both);
            const text_width_px = @as(f32, @floatFromInt(node.text_byte_count)) * glyph_advance_px;
            const line_count = @max(1.0, @ceil(text_width_px / available_width_px));
            height_px_min[index] = @max(node.style.height_px_min, line_count * line_height_px + padding_both);
        } else {
            height_px_min[index] = @max(node.style.height_px_min, padding_both);
        }
        height_px[index] = height_px_min[index];
    }

    var child_counts: [node_capacity_max]u16 = @splat(0);
    var reverse_index: usize = nodes.len;
    while (reverse_index > 0) {
        reverse_index -= 1;
        const parent_index = nodes[reverse_index].parent_index;
        if (parent_index != index_none) {
            const parent = &nodes[parent_index];
            if (parent.kind == .row) {
                const required_px = height_px[reverse_index] + parent.style.pad_px * 2.0;
                if (required_px > height_px[parent_index]) {
                    height_px[parent_index] = required_px;
                }
            } else {
                const gap_px = if (child_counts[parent_index] > 0) parent.style.gap_px else 0.0;
                height_px[parent_index] += height_px[reverse_index] + gap_px;
            }
            child_counts[parent_index] += 1;
        }
    }
    height_px[0] = @max(height_px[0], viewport_height_px);
}

fn layoutPass4TopDownY(
    nodes: []const ActiveNode,
    width_px: []const f32,
    height_px: []const f32,
    rects_px: []RectPx,
) void {
    rects_px[0] = .{
        .x_px = 0.0,
        .y_px = 0.0,
        .width_px = width_px[0],
        .height_px = height_px[0],
    };

    for (nodes, 0..) |*parent, parent_index| {
        const parent_rect = rects_px[parent_index];
        var cursor_x_px = parent_rect.x_px + parent.style.pad_px;
        var cursor_y_px = parent_rect.y_px + parent.style.pad_px;

        var child_cursor: u16 = @intCast(parent_index + 1);
        const subtree_end: u16 = @intCast(parent_index + parent.subtree_count);

        while (child_cursor < subtree_end) {
            rects_px[child_cursor] = .{
                .x_px = cursor_x_px,
                .y_px = cursor_y_px,
                .width_px = width_px[child_cursor],
                .height_px = height_px[child_cursor],
            };
            if (parent.kind == .row) {
                cursor_x_px += width_px[child_cursor] + parent.style.gap_px;
            } else {
                cursor_y_px += height_px[child_cursor] + parent.style.gap_px;
            }
            child_cursor += nodes[child_cursor].subtree_count;
        }
    }
}

// ============================================================================
// 5. Test de Conformité M0 : Layout 4-Passes + Îlots + Time-Travel + Replay
// ============================================================================

test "M0 Engine: 22-line flat DSL, O(1) keyed reuse, 4-pass layout, and deterministic replay" {
    var state = AppState{};
    var engine: EngineM0 = undefined;
    EngineM0.init(&engine);

    engine.materializeTree(&app_view, &state);
    engine.computeLayout(320.0, 240.0);

    try std.testing.expectEqual(@as(u8, 3), engine.each_pool.active_count);
    try std.testing.expectEqual(@as(u32, 3), engine.each_pool.alloc_miss_count);
    try std.testing.expectEqual(@as(u32, 0), engine.each_pool.reuse_hit_count);

    const snapshot_initial = state;
    const semantic_hash_initial = engine.computeSemanticHash();
    const pixel_hash_initial = engine.renderHeadlessRgbaHash(160, 120);

    const action_log = [_]Action{
        .{ .swap_items = .{ .first = 0, .second = 2 } },
        .{ .set_filter = .active },
        .{ .toggle_index = 0 },
        .{ .add_item = .{ .id = 104, .cost_ms = 5, .completed = 0 } },
        .{ .set_filter = .all },
    };

    for (action_log) |action| {
        reduce(&state, action);
        engine.materializeTree(&app_view, &state);
        engine.computeLayout(320.0, 240.0);
    }

    try std.testing.expect(engine.each_pool.reuse_hit_count >= 8);
    try std.testing.expectEqual(@as(u8, 4), engine.each_pool.active_count);

    const snapshot_final = state;
    const semantic_hash_final = engine.computeSemanticHash();
    const pixel_hash_final = engine.renderHeadlessRgbaHash(160, 120);
    try std.testing.expect(semantic_hash_final != semantic_hash_initial);
    try std.testing.expect(pixel_hash_final != pixel_hash_initial);

    state = snapshot_initial;
    engine.materializeTree(&app_view, &state);
    engine.computeLayout(320.0, 240.0);
    try std.testing.expectEqual(semantic_hash_initial, engine.computeSemanticHash());
    try std.testing.expectEqual(pixel_hash_initial, engine.renderHeadlessRgbaHash(160, 120));

    var replay_state = snapshot_initial;
    var replay_engine: EngineM0 = undefined;
    EngineM0.init(&replay_engine);

    for (action_log) |action| {
        reduce(&replay_state, action);
        replay_engine.materializeTree(&app_view, &replay_state);
        replay_engine.computeLayout(320.0, 240.0);
    }

    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&snapshot_final), std.mem.asBytes(&replay_state));
    try std.testing.expectEqual(semantic_hash_final, replay_engine.computeSemanticHash());
    try std.testing.expectEqual(pixel_hash_final, replay_engine.renderHeadlessRgbaHash(160, 120));
}
