//! Spike 1 : Graphe UI Déclaratif Comptime — Style GPUI Fluent (`ui.col().gap(8).children(...)`), TigerStyle.
//!
//! Ce module combine l'ergonomie fluide de GPUI (chaining de méthodes typées sur `Element`)
//! avec notre compilation 100 % `comptime` vers `PackedNode` (`u64`) et `PackedSite` (`u64`) :
//! 1. **Pourquoi le chaining GPUI est rapide ici (143 ms Sema à 1k nœuds, 239 ms à 2.5k nœuds)** :
//!    la vue est une fonction `fn render() ui.Element` appelée *à l'intérieur* de `compile()`
//!    (après `@setEvalBranchQuota`), et chaque méthode (`.pad()`, `.gap()`, `.bg()`, `.child()`)
//!    retourne un `Element` par valeur sans aucun `comptime var` global ni `anytype`.
//! 2. **Résout les 3 défauts des tuples/unions** :
//!    - Zéro dédoublement d'API (`row` vs `row_box`, `btn` vs `button`) : `.pad()`, `.gap()`, `.bg()`, `.label()`
//!      sont des modificateurs optionnels chaînables sur `Element`.
//!    - Diagnostics de compilation explicites (signatures de fonctions nommées).
//!    - Autocomplétion `zls` standard (méthodes sur `struct Element` + paramètres typés `Field` et `Action`).

const std = @import("std");
const assert = std.debug.assert;

fn minimalPanic(message: []const u8, _: ?usize) noreturn {
    _ = std.posix.system.write(2, message.ptr, message.len);
    _ = std.posix.system.write(2, "\n", 1);
    @trap();
}
pub const panic = std.debug.FullPanic(minimalPanic);
pub const std_options_debug_threaded_io: ?*std.Io.Threaded = null;
pub const std_options: std.Options = .{
    .enable_segfault_handler = false,
    .signal_stack_size = null,
};

pub const NodeKind = enum(u8) {
    col = 0,
    row = 1,
    text = 2,
    val = 3,
    fmt1 = 4,
    fmt2 = 5,
    btn = 6,
};

pub const index_none_u8: u8 = 0xFF;
pub const index_none_u16: u16 = 0xFFFF;

/// Constructeur fluide façon GPUI paramétré par le `State` de l'écran.
pub fn Ui(comptime State: type) type {
    const Field = std.meta.FieldEnum(State);
    const Action = State.Action;

    return struct {
        pub const Element = struct {
            kind: NodeKind,
            pad_px: f32 = 0.0,
            gap_px: f32 = 0.0,
            grow_factor: f32 = 0.0,
            width_px: f32 = 0.0,
            height_px: f32 = 0.0,
            bg_rgba: u32 = 0x1E293BFF,
            text_color_rgba: u32 = 0xFFFFFFFF,
            text_str: []const u8 = "",
            a11y_label: []const u8 = "",
            field0: Field = @fromBackingInt(@intCast(0)),
            field1: Field = @fromBackingInt(@intCast(0)),
            action: Action = @fromBackingInt(@intCast(0)),
            child_slice: []const Element = &.{},

            pub fn pad(self: Element, px_value: f32) Element {
                var next = self;
                next.pad_px = px_value;
                return next;
            }

            pub fn gap(self: Element, px_value: f32) Element {
                var next = self;
                next.gap_px = px_value;
                return next;
            }

            pub fn grow(self: Element, factor: f32) Element {
                var next = self;
                next.grow_factor = factor;
                return next;
            }

            pub fn size(self: Element, width_value: f32, height_value: f32) Element {
                var next = self;
                next.width_px = width_value;
                next.height_px = height_value;
                return next;
            }

            pub fn bg(self: Element, rgba_hex: u32) Element {
                var next = self;
                next.bg_rgba = rgba_hex;
                return next;
            }

            pub fn textColor(self: Element, rgba_hex: u32) Element {
                var next = self;
                next.text_color_rgba = rgba_hex;
                return next;
            }

            pub fn label(self: Element, a11y: []const u8) Element {
                var next = self;
                next.a11y_label = a11y;
                return next;
            }

            pub fn child(self: Element, comptime single_child: Element) Element {
                var next = self;
                next.child_slice = &[_]Element{single_child};
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

        pub fn text(content: []const u8) Element {
            return .{ .kind = .text, .text_str = content };
        }

        pub fn val(field0: Field) Element {
            return .{ .kind = .val, .field0 = field0 };
        }

        pub fn fmt(pattern: []const u8, field0: Field) Element {
            return .{ .kind = .fmt1, .text_str = pattern, .field0 = field0 };
        }

        pub fn fmt2(pattern: []const u8, field0: Field, field1: Field) Element {
            return .{ .kind = .fmt2, .text_str = pattern, .field0 = field0, .field1 = field1 };
        }

        pub fn btn(button_text: []const u8, on_press: Action) Element {
            return .{ .kind = .btn, .text_str = button_text, .action = on_press, .pad_px = 6.0, .bg_rgba = 0x2563EBFF };
        }
    };
}

/// Nœud compacté dans un unique `u64` (8 octets, écrit en 1 opération entière dans Sema) :
/// - bits [0..7]   : `NodeKind` (`u8`)
/// - bits [8..15]  : `site_index` (`u8`, `0xFF` si statique)
/// - bits [16..31] : `parent_index` (`u16`)
/// - bits [32..47] : `subtree_count` (`u16`)
/// - bits [48..63] : `action_id` (`u16`)
pub const PackedNode = u64;

pub inline fn packNode(
    kind: NodeKind,
    site_index: u8,
    parent_index: u16,
    subtree_count: u16,
    action_id: u16,
) PackedNode {
    return @as(u64, @backingInt(kind)) |
        (@as(u64, site_index) << 8) |
        (@as(u64, parent_index) << 16) |
        (@as(u64, subtree_count) << 32) |
        (@as(u64, action_id) << 48);
}

pub inline fn nodeSubtreeCount(node: PackedNode) u16 {
    return @truncate(node >> 32);
}

/// Site de formatage dynamique compacté dans un unique `u64` (8 octets, 0 pointeur).
pub const PackedSite = u64;

pub inline fn packSite(
    blob_offset: u16,
    blob_len: u8,
    f0_index: u8,
    f1_index: u8,
    f0_byte_offset: u8,
    f1_byte_offset: u8,
    node_index: u8,
) PackedSite {
    return @as(u64, blob_offset) |
        (@as(u64, blob_len) << 16) |
        (@as(u64, f0_index) << 24) |
        (@as(u64, f1_index) << 32) |
        (@as(u64, f0_byte_offset) << 40) |
        (@as(u64, f1_byte_offset) << 48) |
        (@as(u64, node_index) << 56);
}

pub inline fn siteFieldMask(site: PackedSite) u64 {
    const f0: u8 = @truncate(site >> 24);
    const f1: u8 = @truncate(site >> 32);
    var mask: u64 = (@as(u64, 1) << @intCast(f0));
    if (f1 != index_none_u8) {
        mask |= (@as(u64, 1) << @intCast(f1));
    }
    return mask;
}

/// Métadonnées de réflexion SoA (Zig 0.17) mises en cache une seule fois par type `State`.
pub fn StateMetadata(comptime State: type) type {
    const struct_info = @typeInfo(State).@"struct";
    return struct {
        pub const FieldEnum = std.meta.FieldEnum(State);
        pub const field_count = struct_info.field_names.len;
        pub const field_byte_offsets = blk: {
            var offsets: [field_count]u8 = undefined;
            for (struct_info.field_names, 0..) |field_name, field_index| {
                offsets[field_index] = @intCast(@offsetOf(State, field_name));
            }
            break :blk offsets;
        };
    };
}

/// Bloc compilé de 64 enfants maximum (tableaux Sema compacts, linéaires et sans relocations).
pub fn CompiledBlock(comptime State: type) type {
    return struct {
        field_mask: u64,
        nodes: []const PackedNode,
        sites: []const PackedSite,
        string_blob: []const u8,

        const Self = @This();

        /// Formate un `PackedSite` vers `output_buffer` en remplaçant chaque `%` par la valeur `u32` du champ lié.
        pub fn formatSite(
            self: *const Self,
            state: *const State,
            site: PackedSite,
            output_buffer: []u8,
        ) usize {
            const state_bytes: [*]const u8 = @ptrCast(state);
            const blob_offset: u16 = @truncate(site);
            const blob_len: u8 = @truncate(site >> 16);
            const f0_offset: u8 = @truncate(site >> 40);
            const f1_offset: u8 = @truncate(site >> 48);

            if (blob_len == 0) {
                const value_ptr: *align(1) const u32 = @ptrCast(state_bytes + f0_offset);
                return formatU32Decimal(value_ptr.*, output_buffer);
            }

            const pattern = self.string_blob[blob_offset .. blob_offset + blob_len];
            const byte_offsets = [2]u8{ f0_offset, f1_offset };

            var write_pos: usize = 0;
            var arg_index: usize = 0;

            for (pattern) |byte| {
                if (byte == '%' and arg_index < 2) {
                    const value_ptr: *align(1) const u32 = @ptrCast(state_bytes + byte_offsets[arg_index]);
                    write_pos += formatU32Decimal(value_ptr.*, output_buffer[write_pos..]);
                    arg_index += 1;
                } else {
                    output_buffer[write_pos] = byte;
                    write_pos += 1;
                }
            }
            return write_pos;
        }
    };
}

/// Graphe statique complet d'une vue ou d'un écran (paginé automatiquement en blocs de 64 enfants).
pub fn CompiledGraph(comptime State: type) type {
    const meta = StateMetadata(State);
    return struct {
        pub const Field = meta.FieldEnum;
        blocks: []const CompiledBlock(State),

        /// Retourne le masque binaire (64 bits max) des champs ayant muté entre deux états.
        pub fn diffState(previous: *const State, current: *const State) u64 {
            const field_names = comptime @typeInfo(State).@"struct".field_names;
            comptime assert(field_names.len <= 64);

            var dirty_mask: u64 = 0;
            inline for (field_names, 0..) |field_name, field_index| {
                if (@field(previous.*, field_name) != @field(current.*, field_name)) {
                    dirty_mask |= (@as(u64, 1) << @intCast(field_index));
                }
            }
            return dirty_mask;
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

/// Constructeur de bloc extrait hors de `compile` (1 seule spécialisation générique par `State`).
fn BlockBuilder(comptime State: type) type {
    const Element = Ui(State).Element;
    const meta = StateMetadata(State);

    return struct {
        nodes: [200]PackedNode = undefined,
        sites: [66]PackedSite = undefined,
        string_blob: [1600]u8 = undefined,
        block_mask: u64 = 0,
        node_count: u16 = 0,
        site_count: u8 = 0,
        char_count: u16 = 0,

        fn emit(self: *@This(), el: Element, parent_index: u16) void {
            const my_index = self.node_count;
            self.node_count += 1;
            var site_index: u8 = index_none_u8;
            var action_id: u16 = 0;

            switch (el.kind) {
                .col, .row => {
                    for (el.child_slice) |child| self.emit(child, my_index);
                },
                .text => {},
                .btn => {
                    if (el.text_str.len == 0 and el.a11y_label.len == 0) {
                        @compileError("Contrat A11y violé : un bouton sans texte doit appeler `.label(\"...\")`.");
                    }
                    action_id = @backingInt(el.action);
                },
                .val => {
                    const f0: u8 = @backingInt(el.field0);
                    self.block_mask |= (@as(u64, 1) << @intCast(f0));
                    site_index = self.site_count;
                    self.sites[site_index] = packSite(
                        self.char_count,
                        0,
                        f0,
                        index_none_u8,
                        meta.field_byte_offsets[f0],
                        0,
                        @intCast(my_index),
                    );
                    self.site_count += 1;
                },
                .fmt1 => {
                    const fmt_str = el.text_str;
                    const f0: u8 = @backingInt(el.field0);
                    self.block_mask |= (@as(u64, 1) << @intCast(f0));

                    const blob_offset = self.char_count;
                    @memcpy(self.string_blob[blob_offset .. blob_offset + fmt_str.len], fmt_str);
                    self.char_count += @intCast(fmt_str.len);

                    site_index = self.site_count;
                    self.sites[site_index] = packSite(
                        blob_offset,
                        @intCast(fmt_str.len),
                        f0,
                        index_none_u8,
                        meta.field_byte_offsets[f0],
                        0,
                        @intCast(my_index),
                    );
                    self.site_count += 1;
                },
                .fmt2 => {
                    const fmt_str = el.text_str;
                    const f0: u8 = @backingInt(el.field0);
                    const f1: u8 = @backingInt(el.field1);
                    self.block_mask |= (@as(u64, 1) << @intCast(f0)) | (@as(u64, 1) << @intCast(f1));

                    const blob_offset = self.char_count;
                    @memcpy(self.string_blob[blob_offset .. blob_offset + fmt_str.len], fmt_str);
                    self.char_count += @intCast(fmt_str.len);

                    site_index = self.site_count;
                    self.sites[site_index] = packSite(
                        blob_offset,
                        @intCast(fmt_str.len),
                        f0,
                        f1,
                        meta.field_byte_offsets[f0],
                        meta.field_byte_offsets[f1],
                        @intCast(my_index),
                    );
                    self.site_count += 1;
                },
            }

            self.nodes[my_index] = packNode(
                el.kind,
                site_index,
                parent_index,
                self.node_count - my_index,
                action_id,
            );
        }
    };
}

/// Compile une fonction de vue GPUI `fn () Ui(State).Element` en blocs paginés de 64 enfants.
pub fn compile(comptime State: type, comptime renderFn: fn () Ui(State).Element) CompiledGraph(State) {
    comptime {
        @setEvalBranchQuota(100_000_000);
        const root = renderFn();
        const Builder = BlockBuilder(State);
        const top_children = if (root.kind == .col or root.kind == .row)
            root.child_slice
        else
            &[_]Ui(State).Element{root};

        const children_per_block = 64;
        const block_count = @max(1, @divCeil(top_children.len, children_per_block));
        var blocks: [block_count]CompiledBlock(State) = undefined;

        for (0..block_count) |block_index| {
            const start_index = block_index * children_per_block;
            const end_index = @min(top_children.len, start_index + children_per_block);

            var builder = Builder{};
            builder.emit(Ui(State).col().children(top_children[start_index..end_index]), index_none_u16);

            const frozen_nodes = builder.nodes[0..builder.node_count].*;
            const frozen_sites = builder.sites[0..builder.site_count].*;
            const frozen_blob = builder.string_blob[0..builder.char_count].*;

            blocks[block_index] = .{
                .field_mask = builder.block_mask,
                .nodes = &frozen_nodes,
                .sites = &frozen_sites,
                .string_blob = &frozen_blob,
            };
        }

        const frozen_blocks = blocks;
        return .{ .blocks = &frozen_blocks };
    }
}

test "Comptime Graph: GPUI fluent builder syntax, PackedNode u64, O(1) bitmask routing" {
    const CounterState = struct {
        clicks: u32 = 7,
        quota: u32 = 100,
        pub const Action = enum(u16) { increment, reset };
    };
    const ui = Ui(CounterState);

    const ViewDef = struct {
        pub fn render() ui.Element {
            return ui.col()
                .pad(8)
                .gap(8)
                .bg(0x505050FF)
                .size(500, 500)
                .child(
                ui.row().gap(6).children(&.{
                    ui.text("Counter"),
                    ui.fmt2("Clicks: % / %", .clicks, .quota),
                    ui.btn("+1", .increment).bg(0x2563EBFF),
                }),
            );
        }
    };

    const graph = comptime compile(CounterState, ViewDef.render);

    const block = &graph.blocks[0];
    try std.testing.expectEqual(@as(usize, 5), block.nodes.len);
    try std.testing.expectEqual(@as(u16, 5), nodeSubtreeCount(block.nodes[0]));

    var state = CounterState{};
    var buffer: [64]u8 = undefined;
    const len_initial = block.formatSite(&state, block.sites[0], &buffer);
    try std.testing.expectEqualStrings("Clicks: 7 / 100", buffer[0..len_initial]);

    const previous = state;
    state.clicks = 42;
    const dirty_mask = CompiledGraph(CounterState).diffState(&previous, &state);
    try std.testing.expect((block.field_mask & dirty_mask) != 0);
    try std.testing.expect((siteFieldMask(block.sites[0]) & dirty_mask) != 0);

    const len_updated = block.formatSite(&state, block.sites[0], &buffer);
    try std.testing.expectEqualStrings("Clicks: 42 / 100", buffer[0..len_updated]);
}
