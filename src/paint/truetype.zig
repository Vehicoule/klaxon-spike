//! ============================================================================
//! Klaxon TrueType (.ttf) Font Decoder (`src/paint/truetype.zig`)
//! ============================================================================
//! Zero-libc, zero-heap TrueType table parser & outline geometry decoder:
//! 1. Parses 7 essential TrueType tables:
//!    - 'head': Units per em, loca offset format (short u16*2 or long u32), font bbox.
//!    - 'maxp': Number of glyphs, maximum points and contours.
//!    - 'loca': Glyph offset index table.
//!    - 'glyf': Vectorial outline geometry (simple and composite glyphs).
//!    - 'cmap': Character codepoint to glyph ID mapping (Formats 4, 12, 0).
//!    - 'hhea': Horizontal metrics header (ascender, descender, line gap).
//!    - 'hmtx': Horizontal advance widths and left side bearings.
//! 2. Implicit on-curve midpoints: Resolves consecutive off-curve control points.
//! 3. Composite glyphs: Recursive component decomposition with 2D affine transforms.
//! 4. Strict bounds checking on all table reads; returns typed errors.
//!
//! Standards: TigerStyle (<= 100 columns, zero libc, zero POSIX, 0 malloc/free).

const std = @import("std");
const assert = std.debug.assert;
const sdf = @import("sdf.zig");
const platform = @import("../platform/os.zig");

pub const Point = sdf.Point;
pub const BezierQuadratic = sdf.BezierQuadratic;
pub const lineArc = sdf.lineArc;
pub const quadArc = sdf.quadArc;

// ============================================================================
// TrueType Big-Endian Binary Readers
// ============================================================================

pub inline fn readU16(data: []const u8, offset: usize) u16 {
    assert(offset + 2 <= data.len);
    return std.mem.readInt(u16, data[offset..][0..2], .big);
}

pub inline fn readI16(data: []const u8, offset: usize) i16 {
    assert(offset + 2 <= data.len);
    return std.mem.readInt(i16, data[offset..][0..2], .big);
}

pub inline fn readU32(data: []const u8, offset: usize) u32 {
    assert(offset + 4 <= data.len);
    return std.mem.readInt(u32, data[offset..][0..4], .big);
}

pub inline fn readI32(data: []const u8, offset: usize) i32 {
    assert(offset + 4 <= data.len);
    return std.mem.readInt(i32, data[offset..][0..4], .big);
}

// ============================================================================
// Metrics & Statistics Data Structures
// ============================================================================

/// Horizontal layout metrics for a specific glyph.
pub const GlyphMetrics = extern struct {
    advance_width: u16 = 0,
    left_side_bearing: i16 = 0,
};

comptime {
    assert(@sizeOf(GlyphMetrics) == 4);
    assert(@alignOf(GlyphMetrics) == 2);
}

/// Geometry decomposition statistics.
pub const GlyphStats = struct {
    lines_count: usize = 0,
    quads_count: usize = 0,
    implicit_midpoints: usize = 0,
    composites_count: usize = 0,
};

/// Working buffers for glyph outline decoding without heap allocations.
pub const ScratchBuffers = struct {
    points: [2048]Point = undefined,
    end_pts: [128]u16 = undefined,
    flags: [2048]u8 = undefined,
};

/// Typed errors for TrueType font parsing and validation.
pub const FontError = error{
    InvalidFontData,
    TruncatedTableDirectory,
    RequiredTableMissing,
    InvalidTableOffset,
    InvalidGlyphIndex,
    UnsupportedLocaFormat,
    BufferTooSmall,
    RecursionLimitExceeded,
};

// ============================================================================
// TrueType Font Parser (`TrueTypeFont`)
// ============================================================================

pub const TrueTypeFont = struct {
    data: []const u8,

    // Parsed Table Offsets
    head_offset: usize = 0,
    maxp_offset: usize = 0,
    loca_offset: usize = 0,
    glyf_offset: usize = 0,
    cmap_offset: usize = 0,
    hhea_offset: usize = 0,
    hmtx_offset: usize = 0,

    // Header ('head')
    units_per_em: u16 = 1000,
    loc_format: i16 = 0,
    bbox_min: [2]i16 = .{ 0, 0 },
    bbox_max: [2]i16 = .{ 0, 0 },

    // Maxp ('maxp')
    num_glyphs: u16 = 0,

    // Horizontal Header ('hhea')
    ascender: i16 = 0,
    descender: i16 = 0,
    line_gap: i16 = 0,
    advance_width_max: u16 = 0,
    num_h_metrics: u16 = 0,

    /// Initializes and validates a TrueType font buffer in-place.
    pub fn init(data: []const u8) FontError!TrueTypeFont {
        if (data.len < 12) return error.InvalidFontData;

        const sfnt_version = readU32(data, 0);
        // Valid SFNT version tags: 0x00010000 (TrueType 1.0), 'true', 'typ1', 'OTTO'
        const is_tt = (sfnt_version == 0x00010000 or
            sfnt_version == 0x74727565 or
            sfnt_version == 0x74797031 or
            sfnt_version == 0x4F54544F);
        if (!is_tt) return error.InvalidFontData;

        const num_tables = readU16(data, 4);
        var head_offset: usize = 0;
        var maxp_offset: usize = 0;
        var loca_offset: usize = 0;
        var glyf_offset: usize = 0;
        var cmap_offset: usize = 0;
        var hhea_offset: usize = 0;
        var hmtx_offset: usize = 0;

        var i: usize = 0;
        while (i < num_tables) : (i += 1) {
            const entry_offset = 12 + i * 16;
            if (entry_offset + 16 > data.len) return error.TruncatedTableDirectory;
            const tag = data[entry_offset .. entry_offset + 4];
            const offset = readU32(data, entry_offset + 8);
            const length = readU32(data, entry_offset + 12);

            if (@as(u64, offset) + @as(u64, length) > data.len) return error.InvalidTableOffset;

            if (std.mem.eql(u8, tag, "head")) head_offset = offset;
            if (std.mem.eql(u8, tag, "maxp")) maxp_offset = offset;
            if (std.mem.eql(u8, tag, "loca")) loca_offset = offset;
            if (std.mem.eql(u8, tag, "glyf")) glyf_offset = offset;
            if (std.mem.eql(u8, tag, "cmap")) cmap_offset = offset;
            if (std.mem.eql(u8, tag, "hhea")) hhea_offset = offset;
            if (std.mem.eql(u8, tag, "hmtx")) hmtx_offset = offset;
        }

        // Essential geometric tables must be present
        if (head_offset == 0 or maxp_offset == 0 or loca_offset == 0 or glyf_offset == 0) {
            return error.RequiredTableMissing;
        }

        // Validate 'head' table (minimum 54 bytes)
        if (head_offset + 54 > data.len) return error.InvalidTableOffset;
        const magic_number = readU32(data, head_offset + 12);
        if (magic_number != 0x5F0F3CF5) return error.InvalidFontData;

        const units_per_em = readU16(data, head_offset + 18);
        const x_min = readI16(data, head_offset + 36);
        const y_min = readI16(data, head_offset + 38);
        const x_max = readI16(data, head_offset + 40);
        const y_max = readI16(data, head_offset + 42);
        const loc_format = readI16(data, head_offset + 50);

        if (loc_format != 0 and loc_format != 1) return error.UnsupportedLocaFormat;

        // Validate 'maxp' table (minimum 6 bytes)
        if (maxp_offset + 6 > data.len) return error.InvalidTableOffset;
        const num_glyphs = readU16(data, maxp_offset + 4);

        // Parse 'hhea' table if present (minimum 36 bytes)
        var ascender: i16 = 0;
        var descender: i16 = 0;
        var line_gap: i16 = 0;
        var advance_width_max: u16 = 0;
        var num_h_metrics: u16 = 0;

        if (hhea_offset > 0 and hhea_offset + 36 <= data.len) {
            ascender = readI16(data, hhea_offset + 4);
            descender = readI16(data, hhea_offset + 6);
            line_gap = readI16(data, hhea_offset + 8);
            advance_width_max = readU16(data, hhea_offset + 10);
            num_h_metrics = readU16(data, hhea_offset + 34);
        }

        return .{
            .data = data,
            .head_offset = head_offset,
            .maxp_offset = maxp_offset,
            .loca_offset = loca_offset,
            .glyf_offset = glyf_offset,
            .cmap_offset = cmap_offset,
            .hhea_offset = hhea_offset,
            .hmtx_offset = hmtx_offset,
            .units_per_em = units_per_em,
            .loc_format = loc_format,
            .bbox_min = .{ x_min, y_min },
            .bbox_max = .{ x_max, y_max },
            .num_glyphs = num_glyphs,
            .ascender = ascender,
            .descender = descender,
            .line_gap = line_gap,
            .advance_width_max = advance_width_max,
            .num_h_metrics = num_h_metrics,
        };
    }

    /// Resolves glyph byte range relative to the 'glyf' table.
    pub fn getGlyphOffset(self: *const TrueTypeFont, g: u16) struct { o1: usize, o2: usize } {
        if (g >= self.num_glyphs) return .{ .o1 = 0, .o2 = 0 };

        if (self.loc_format == 0) {
            const entry1 = self.loca_offset + @as(usize, g) * 2;
            const entry2 = self.loca_offset + @as(usize, g + 1) * 2;
            if (entry2 + 2 > self.data.len) return .{ .o1 = 0, .o2 = 0 };
            const o1 = @as(usize, readU16(self.data, entry1)) * 2;
            const o2 = @as(usize, readU16(self.data, entry2)) * 2;
            return .{ .o1 = o1, .o2 = o2 };
        } else {
            const entry1 = self.loca_offset + @as(usize, g) * 4;
            const entry2 = self.loca_offset + @as(usize, g + 1) * 4;
            if (entry2 + 4 > self.data.len) return .{ .o1 = 0, .o2 = 0 };
            const o1 = readU32(self.data, entry1);
            const o2 = readU32(self.data, entry2);
            return .{ .o1 = o1, .o2 = o2 };
        }
    }

    /// Resolves horizontal metrics (advance width & left side bearing) for a glyph.
    pub fn getGlyphMetrics(self: *const TrueTypeFont, glyph_idx: u16) GlyphMetrics {
        if (self.hmtx_offset == 0 or self.num_h_metrics == 0) {
            return .{ .advance_width = self.units_per_em, .left_side_bearing = 0 };
        }

        if (glyph_idx < self.num_h_metrics) {
            const entry = self.hmtx_offset + @as(usize, glyph_idx) * 4;
            if (entry + 4 <= self.data.len) {
                const adv = readU16(self.data, entry);
                const lsb = readI16(self.data, entry + 2);
                return .{ .advance_width = adv, .left_side_bearing = lsb };
            }
        } else if (glyph_idx < self.num_glyphs) {
            // Remaining glyphs share the advance width of the last metric entry
            const last_adv_entry = self.hmtx_offset + @as(usize, self.num_h_metrics - 1) * 4;
            const lsb_entry = self.hmtx_offset +
                @as(usize, self.num_h_metrics) * 4 +
                @as(usize, glyph_idx - self.num_h_metrics) * 2;
            if (last_adv_entry + 2 <= self.data.len and lsb_entry + 2 <= self.data.len) {
                const adv = readU16(self.data, last_adv_entry);
                const lsb = readI16(self.data, lsb_entry);
                return .{ .advance_width = adv, .left_side_bearing = lsb };
            }
        }

        return .{ .advance_width = self.units_per_em, .left_side_bearing = 0 };
    }

    /// Maps a Unicode codepoint to a glyph index via the 'cmap' table.
    /// Returns 0 (.notdef) if the mapping is absent.
    pub fn glyphIndexForCodepoint(self: *const TrueTypeFont, codepoint: u21) u16 {
        if (self.cmap_offset == 0) return 0;
        if (self.cmap_offset + 4 > self.data.len) return 0;

        const num_tables = readU16(self.data, self.cmap_offset + 2);
        var subtable_format4_offset: usize = 0;
        var subtable_format12_offset: usize = 0;
        var subtable_format0_offset: usize = 0;

        var i: usize = 0;
        while (i < num_tables) : (i += 1) {
            const rec_offset = self.cmap_offset + 4 + i * 8;
            if (rec_offset + 8 > self.data.len) break;

            const platform_id = readU16(self.data, rec_offset);
            const encoding_id = readU16(self.data, rec_offset + 2);
            const offset_val = readU32(self.data, rec_offset + 4);
            const sub_offset = self.cmap_offset + offset_val;

            if (sub_offset + 2 > self.data.len) continue;
            const fmt = readU16(self.data, sub_offset);

            if (fmt == 12) {
                if (platform_id == 0 or (platform_id == 3 and encoding_id == 10)) {
                    subtable_format12_offset = sub_offset;
                }
            } else if (fmt == 4) {
                const is_p0 = (platform_id == 0);
                const is_p3 = (platform_id == 3 and (encoding_id == 1 or encoding_id == 0));
                if (is_p0 or is_p3) {
                    if (subtable_format4_offset == 0 or platform_id == 3) {
                        subtable_format4_offset = sub_offset;
                    }
                }
            } else if (fmt == 0 and subtable_format0_offset == 0) {
                subtable_format0_offset = sub_offset;
            }
        }

        // Format 12: 32-bit full Unicode coverage
        if (subtable_format12_offset > 0 and subtable_format12_offset + 16 <= self.data.len) {
            const num_groups = readU32(self.data, subtable_format12_offset + 12);
            var g: usize = 0;
            while (g < num_groups) : (g += 1) {
                const grp_offset = subtable_format12_offset + 16 + g * 12;
                if (grp_offset + 12 > self.data.len) break;
                const start_char = readU32(self.data, grp_offset);
                const end_char = readU32(self.data, grp_offset + 4);
                const start_glyph = readU32(self.data, grp_offset + 8);

                if (codepoint >= start_char and codepoint <= end_char) {
                    const mapped = start_glyph + (codepoint - start_char);
                    return @intCast(@min(mapped, std.math.maxInt(u16)));
                }
            }
        }

        // Format 4: 16-bit BMP segment mapping
        if (subtable_format4_offset > 0 and codepoint <= 0xFFFF) {
            const sub = subtable_format4_offset;
            if (sub + 14 <= self.data.len) {
                const seg_count_x2 = readU16(self.data, sub + 6);
                const seg_count = seg_count_x2 / 2;

                const end_code_offset = sub + 14;
                const start_code_offset = end_code_offset + seg_count * 2 + 2;
                const id_delta_offset = start_code_offset + seg_count * 2;
                const id_range_offset = id_delta_offset + seg_count * 2;

                if (id_range_offset + seg_count * 2 <= self.data.len) {
                    const cp16: u16 = @intCast(codepoint);
                    var s: usize = 0;
                    while (s < seg_count) : (s += 1) {
                        const end_code = readU16(self.data, end_code_offset + s * 2);
                        if (cp16 <= end_code) {
                            const start_code = readU16(self.data, start_code_offset + s * 2);
                            if (cp16 >= start_code) {
                                const id_delta = readI16(self.data, id_delta_offset + s * 2);
                                const range_val = readU16(self.data, id_range_offset + s * 2);

                                if (range_val == 0) {
                                    return cp16 +% @as(u16, @bitCast(id_delta));
                                } else {
                                    const range_entry_offset = id_range_offset + s * 2;
                                    const glyph_ptr = range_entry_offset + range_val +
                                        @as(usize, cp16 - start_code) * 2;
                                    if (glyph_ptr + 2 <= self.data.len) {
                                        const raw_id = readU16(self.data, glyph_ptr);
                                        if (raw_id != 0) {
                                            return raw_id +% @as(u16, @bitCast(id_delta));
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Format 0: Simple 256-byte table
        if (subtable_format0_offset > 0 and codepoint < 256) {
            const entry = subtable_format0_offset + 6 + codepoint;
            if (entry < self.data.len) {
                return @as(u16, self.data[entry]);
            }
        }

        return 0;
    }

    /// Decodes Bézier curves of a glyph into out_curves using caller-provided scratch buffers.
    pub fn decodeGlyphCurves(
        self: *const TrueTypeFont,
        glyph_idx: u16,
        scratch: *ScratchBuffers,
        out_curves: []BezierQuadratic,
        curve_count: *usize,
        stats: *GlyphStats,
        depth: u32,
    ) void {
        if (depth > 4) return;
        const offsets = self.getGlyphOffset(glyph_idx);
        if (offsets.o1 >= offsets.o2) return;
        if (offsets.o2 > self.data.len) return;
        if (@as(u64, self.glyf_offset) + offsets.o2 > self.data.len) return;

        const start = self.glyf_offset + offsets.o1;
        const end = self.glyf_offset + offsets.o2;
        if (start >= end or end > self.data.len) return;

        const g_data = self.data[start..end];
        if (g_data.len < 10) return;

        const num_contours = readI16(g_data, 0);
        if (num_contours >= 0) {
            self.decodeSimpleGlyph(g_data, scratch, out_curves, curve_count, stats);
        } else {
            self.decodeCompositeGlyph(g_data, scratch, out_curves, curve_count, stats, depth);
        }
    }

    fn decodeSimpleGlyph(
        self: *const TrueTypeFont,
        g_data: []const u8,
        scratch: *ScratchBuffers,
        out_curves: []BezierQuadratic,
        curve_count: *usize,
        stats: *GlyphStats,
    ) void {
        _ = self;
        const num_contours_i = readI16(g_data, 0);
        if (num_contours_i <= 0) return;
        const num_contours = @as(usize, @intCast(num_contours_i));
        if (num_contours > scratch.end_pts.len) return;

        const end_pts_header = 10 + num_contours * 2;
        if (end_pts_header + 2 > g_data.len) return;

        var c: usize = 0;
        while (c < num_contours) : (c += 1) {
            scratch.end_pts[c] = readU16(g_data, 10 + c * 2);
        }

        const num_points = @as(usize, scratch.end_pts[num_contours - 1]) + 1;
        if (num_points > scratch.points.len) return;

        const instr_len = readU16(g_data, 10 + num_contours * 2);
        var offset = 12 + num_contours * 2 + instr_len;
        if (offset > g_data.len) return;

        // 1. Decode flags with repeat compression (0x08)
        var flag_idx: usize = 0;
        while (flag_idx < num_points and offset < g_data.len) {
            const flag = g_data[offset];
            offset += 1;
            scratch.flags[flag_idx] = flag;
            flag_idx += 1;

            if ((flag & 0x08) != 0 and offset < g_data.len) {
                const repeat_count = g_data[offset];
                offset += 1;
                var r: usize = 0;
                while (r < repeat_count and flag_idx < num_points) : (r += 1) {
                    scratch.flags[flag_idx] = flag;
                    flag_idx += 1;
                }
            }
        }
        if (flag_idx < num_points) return;

        // 2. Decode delta-X coordinates
        var cur_x: i32 = 0;
        var p_idx: usize = 0;
        while (p_idx < num_points and offset <= g_data.len) : (p_idx += 1) {
            const flag = scratch.flags[p_idx];
            var dx: i32 = 0;
            if ((flag & 0x02) != 0) {
                if (offset >= g_data.len) break;
                const val = @as(i32, g_data[offset]);
                offset += 1;
                dx = if ((flag & 0x10) != 0) val else -val;
            } else {
                if ((flag & 0x10) != 0) {
                    dx = 0;
                } else {
                    if (offset + 2 > g_data.len) break;
                    dx = readI16(g_data, offset);
                    offset += 2;
                }
            }
            cur_x += dx;
            scratch.points[p_idx].x = @as(f32, @floatFromInt(cur_x));
            scratch.points[p_idx].on_curve = ((flag & 0x01) != 0);
        }
        if (p_idx < num_points) return;

        // 3. Decode delta-Y coordinates
        var cur_y: i32 = 0;
        p_idx = 0;
        while (p_idx < num_points and offset <= g_data.len) : (p_idx += 1) {
            const flag = scratch.flags[p_idx];
            var dy: i32 = 0;
            if ((flag & 0x04) != 0) {
                if (offset >= g_data.len) break;
                const val = @as(i32, g_data[offset]);
                offset += 1;
                dy = if ((flag & 0x20) != 0) val else -val;
            } else {
                if ((flag & 0x20) != 0) {
                    dy = 0;
                } else {
                    if (offset + 2 > g_data.len) break;
                    dy = readI16(g_data, offset);
                    offset += 2;
                }
            }
            cur_y += dy;
            scratch.points[p_idx].y = @as(f32, @floatFromInt(cur_y));
        }
        if (p_idx < num_points) return;

        // 4. Decompose contours into straight lines and quadratic Bézier arcs
        var start_pt_idx: usize = 0;
        c = 0;
        while (c < num_contours) : (c += 1) {
            const end_pt_idx = @as(usize, scratch.end_pts[c]);
            if (end_pt_idx >= start_pt_idx and end_pt_idx < num_points) {
                const contour = scratch.points[start_pt_idx .. end_pt_idx + 1];
                decomposeContour(contour, out_curves, curve_count, stats);
                start_pt_idx = end_pt_idx + 1;
            }
        }
    }

    fn decodeCompositeGlyph(
        self: *const TrueTypeFont,
        g_data: []const u8,
        scratch: *ScratchBuffers,
        out_curves: []BezierQuadratic,
        curve_count: *usize,
        stats: *GlyphStats,
        depth: u32,
    ) void {
        stats.composites_count += 1;
        var offset: usize = 10;

        while (offset + 4 <= g_data.len) {
            const flags = readU16(g_data, offset);
            const sub_idx = readU16(g_data, offset + 2);
            offset += 4;

            var dx: f32 = 0.0;
            var dy: f32 = 0.0;
            if ((flags & 0x0001) != 0) {
                if (offset + 4 > g_data.len) break;
                const arg1 = readI16(g_data, offset);
                const arg2 = readI16(g_data, offset + 2);
                offset += 4;
                if ((flags & 0x0002) != 0) {
                    dx = @as(f32, @floatFromInt(arg1));
                    dy = @as(f32, @floatFromInt(arg2));
                }
            } else {
                if (offset + 2 > g_data.len) break;
                const arg1 = @as(i8, @bitCast(g_data[offset]));
                const arg2 = @as(i8, @bitCast(g_data[offset + 1]));
                offset += 2;
                if ((flags & 0x0002) != 0) {
                    dx = @as(f32, @floatFromInt(arg1));
                    dy = @as(f32, @floatFromInt(arg2));
                }
            }

            var a: f32 = 1.0;
            var b: f32 = 0.0;
            var c: f32 = 0.0;
            var d: f32 = 1.0;

            if ((flags & 0x0008) != 0) {
                if (offset + 2 > g_data.len) break;
                const scale = @as(f32, @floatFromInt(readI16(g_data, offset))) / 16384.0;
                offset += 2;
                a = scale;
                d = scale;
            } else if ((flags & 0x0040) != 0) {
                if (offset + 4 > g_data.len) break;
                a = @as(f32, @floatFromInt(readI16(g_data, offset))) / 16384.0;
                d = @as(f32, @floatFromInt(readI16(g_data, offset + 2))) / 16384.0;
                offset += 4;
            } else if ((flags & 0x0080) != 0) {
                if (offset + 8 > g_data.len) break;
                a = @as(f32, @floatFromInt(readI16(g_data, offset))) / 16384.0;
                b = @as(f32, @floatFromInt(readI16(g_data, offset + 2))) / 16384.0;
                c = @as(f32, @floatFromInt(readI16(g_data, offset + 4))) / 16384.0;
                d = @as(f32, @floatFromInt(readI16(g_data, offset + 6))) / 16384.0;
                offset += 8;
            }

            // Recursive decoding of sub-component
            const start_curve_idx = curve_count.*;
            self.decodeGlyphCurves(sub_idx, scratch, out_curves, curve_count, stats, depth + 1);
            const end_curve_idx = curve_count.*;

            // Apply 2D affine transformation to new curves
            var ci = start_curve_idx;
            while (ci < end_curve_idx) : (ci += 1) {
                const crv = out_curves[ci];
                out_curves[ci] = .{
                    .p0 = .{
                        a * crv.p0[0] + c * crv.p0[1] + dx,
                        b * crv.p0[0] + d * crv.p0[1] + dy,
                    },
                    .p1 = .{
                        a * crv.p1[0] + c * crv.p1[1] + dx,
                        b * crv.p1[0] + d * crv.p1[1] + dy,
                    },
                    .p2 = .{
                        a * crv.p2[0] + c * crv.p2[1] + dx,
                        b * crv.p2[0] + d * crv.p2[1] + dy,
                    },
                };
            }

            if ((flags & 0x0020) == 0) break;
        }
    }
};

/// Decomposes points into Bézier segments with implicit midpoint generation.
pub fn decomposeContour(
    contour_pts: []const Point,
    out_curves: []BezierQuadratic,
    curve_count: *usize,
    stats: *GlyphStats,
) void {
    const m = contour_pts.len;
    if (m < 2) return;

    var start_idx: usize = 0;
    var start_pt: [2]f32 = undefined;

    if (contour_pts[0].on_curve) {
        start_pt = .{ contour_pts[0].x, contour_pts[0].y };
        start_idx = 1;
    } else if (contour_pts[m - 1].on_curve) {
        start_pt = .{ contour_pts[m - 1].x, contour_pts[m - 1].y };
        start_idx = 0;
    } else {
        // Two consecutive off-curve endpoints -> implicit on-curve midpoint
        start_pt = .{
            (contour_pts[0].x + contour_pts[m - 1].x) * 0.5,
            (contour_pts[0].y + contour_pts[m - 1].y) * 0.5,
        };
        start_idx = 0;
        stats.implicit_midpoints += 1;
    }

    var curr_pt = start_pt;
    var k: usize = 0;
    while (k < m) {
        const idx = (start_idx + k) % m;
        const pt = contour_pts[idx];

        if (pt.on_curve) {
            const end_pt = [2]f32{ pt.x, pt.y };
            if (curve_count.* < out_curves.len) {
                out_curves[curve_count.*] = lineArc(curr_pt, end_pt);
                curve_count.* += 1;
                stats.lines_count += 1;
            }
            curr_pt = end_pt;
            k += 1;
        } else {
            const ctrl_pt = [2]f32{ pt.x, pt.y };
            k += 1;
            if (k < m) {
                const next_idx = (start_idx + k) % m;
                const next_pt = contour_pts[next_idx];
                if (next_pt.on_curve) {
                    const end_pt = [2]f32{ next_pt.x, next_pt.y };
                    if (curve_count.* < out_curves.len) {
                        out_curves[curve_count.*] = quadArc(curr_pt, ctrl_pt, end_pt);
                        curve_count.* += 1;
                        stats.quads_count += 1;
                    }
                    curr_pt = end_pt;
                    k += 1;
                } else {
                    // Two consecutive off-curve points: implicit on-curve midpoint
                    const end_pt = [2]f32{
                        (ctrl_pt[0] + next_pt.x) * 0.5,
                        (ctrl_pt[1] + next_pt.y) * 0.5,
                    };
                    if (curve_count.* < out_curves.len) {
                        out_curves[curve_count.*] = quadArc(curr_pt, ctrl_pt, end_pt);
                        curve_count.* += 1;
                        stats.quads_count += 1;
                        stats.implicit_midpoints += 1;
                    }
                    curr_pt = end_pt;
                    // Do not increment k so next_pt becomes the next control point
                }
            } else {
                if (curve_count.* < out_curves.len) {
                    out_curves[curve_count.*] = quadArc(curr_pt, ctrl_pt, start_pt);
                    curve_count.* += 1;
                    stats.quads_count += 1;
                }
                curr_pt = start_pt;
            }
        }
    }

    // Close contour if endpoints do not coincide
    if (curr_pt[0] != start_pt[0] or curr_pt[1] != start_pt[1]) {
        if (curve_count.* < out_curves.len) {
            out_curves[curve_count.*] = lineArc(curr_pt, start_pt);
            curve_count.* += 1;
            stats.lines_count += 1;
        }
    }
}

// ============================================================================
// Multi-OS Zero-Libc Font Loading Helper
// ============================================================================

pub const candidate_font_paths = [_][]const u8{
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/local/share/fonts/dejavu/DejaVuSans.ttf",
    "/system/fonts/Roboto-Regular.ttf",
};

/// Loads a TrueType font file directly into caller-provided buffer via platform I/O.
pub fn loadFontFromFile(path: []const u8, buffer: []u8) !usize {
    return platform.readFileToBuffer(path, buffer);
}

/// Discovers and loads the first existing system TrueType font into caller buffer.
pub fn findAndLoadSystemFont(buffer: []u8) !struct { path: []const u8, len: usize } {
    for (candidate_font_paths) |cand| {
        if (loadFontFromFile(cand, buffer)) |len| {
            return .{ .path = cand, .len = len };
        } else |_| {}
    }
    return error.FileNotFound;
}

// ============================================================================
// Unit Tests
// ============================================================================

test "TrueTypeFont validation on corrupted/truncated input" {
    var tiny_buf: [8]u8 = @splat(0);
    try std.testing.expectError(error.InvalidFontData, TrueTypeFont.init(&tiny_buf));

    var header_buf: [16]u8 = @splat(0);
    // Invalid sfnt version
    try std.testing.expectError(error.InvalidFontData, TrueTypeFont.init(&header_buf));
}

test "TrueTypeFont decomposeContour implicit midpoints" {
    const pts = [_]Point{
        .{ .x = 0.0, .y = 0.0, .on_curve = false },
        .{ .x = 10.0, .y = 20.0, .on_curve = false },
        .{ .x = 20.0, .y = 0.0, .on_curve = true },
    };

    var curves_buf: [16]BezierQuadratic = undefined;
    var curve_count: usize = 0;
    var stats: GlyphStats = .{};

    decomposeContour(&pts, &curves_buf, &curve_count, &stats);
    try std.testing.expect(curve_count > 0);
    try std.testing.expect(stats.implicit_midpoints > 0);
}

test "TrueTypeFont real system font parsing when available" {
    var font_buf: [1024 * 1024]u8 align(64) = undefined;
    const loaded = findAndLoadSystemFont(&font_buf) catch return error.SkipZigTest;

    const font = try TrueTypeFont.init(font_buf[0..loaded.len]);
    try std.testing.expect(font.num_glyphs > 50);
    try std.testing.expect(font.units_per_em > 0);
    try std.testing.expect(font.ascender != 0 or font.descender != 0);

    // Test cmap mapping for ASCII 'A' (codepoint 65)
    const glyph_a = font.glyphIndexForCodepoint('A');
    try std.testing.expect(glyph_a > 0);

    // Test horizontal metrics
    const metrics_a = font.getGlyphMetrics(glyph_a);
    try std.testing.expect(metrics_a.advance_width > 0);

    // Test curve decoding
    var scratch = ScratchBuffers{};
    var curves: [256]BezierQuadratic = undefined;
    var count: usize = 0;
    var stats = GlyphStats{};

    font.decodeGlyphCurves(glyph_a, &scratch, &curves, &count, &stats, 0);
    try std.testing.expect(count > 0);
    try std.testing.expect(stats.lines_count > 0 or stats.quads_count > 0);
}

test "TrueTypeFont decodeGlyphCurves truncated glyph bounds safety" {
    var font: TrueTypeFont = undefined;
    font.num_glyphs = 1;
    font.loc_format = 1;
    font.loca_offset = 0;
    font.glyf_offset = 8;

    var font_data = [_]u8{
        0x00, 0x00, 0x00, 0x00, // loca[0] = 0
        0x00, 0x00, 0x00, 0x0A, // loca[1] = 10
        0x00, 0x01, // num_contours = 1
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // bbox (8 bytes)
    };
    font.data = &font_data;

    var scratch = ScratchBuffers{};
    var curves: [16]BezierQuadratic = undefined;
    var count: usize = 0;
    var stats = GlyphStats{};

    font.decodeGlyphCurves(0, &scratch, &curves, &count, &stats, 0);
    try std.testing.expectEqual(@as(usize, 0), count);
}

test "TrueTypeFont cmap Format 12 subtable Unicode platform mapping" {
    // 16-byte dummy prefix so cmap starts at offset 16
    var cmap_data = [_]u8{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // cmap header at offset 16:
        0x00, 0x00, // version = 0
        0x00, 0x01, // num_tables = 1
        // Encoding record: platform_id=0 (Unicode), encoding_id=4, offset=12
        0x00, 0x00,
        0x00, 0x04,
        0x00, 0x00,
        0x00, 0x0C,
        // Format 12 Subtable at sub_offset = 16 + 12 = 28:
        0x00, 0x0C, // format = 12
        0x00, 0x00, // reserved
        0x00, 0x00, 0x00, 0x1C, // length = 28
        0x00, 0x00, 0x00, 0x00, // language = 0
        0x00, 0x00, 0x00, 0x01, // num_groups = 1
        // Group 0: start_char = 0x1F600, end_char = 0x1F600, start_glyph = 42
        0x00, 0x01, 0xF6, 0x00,
        0x00, 0x01, 0xF6, 0x00,
        0x00, 0x00, 0x00, 0x2A,
    };

    var font: TrueTypeFont = undefined;
    font.data = &cmap_data;
    font.cmap_offset = 16;

    const gid = font.glyphIndexForCodepoint(0x1F600);
    try std.testing.expectEqual(@as(u16, 42), gid);
}
