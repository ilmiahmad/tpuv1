// One iDMA transfer between memory spaces, with legal-by-construction constraints.

typedef enum { SP_DDR, SP_UB, SP_ACC, SP_WFIFO } mem_space_e;

class dma_xfer extends uvm_sequence_item;
  rand mem_space_e  src_sp, dst_sp;
  rand int unsigned src_off, dst_off;     // byte offsets inside the space's test region
  rand int unsigned len, reps, sstride, dstride;
  rand bit          src_hi;               // DDR source above 4 GiB
  int unsigned      id;

  // test regions: DDR src and dst never overlap; on-chip memories use one shared region so that
  // later transfers read what earlier ones wrote (multi-hop chains get checked end to end)
  static function longint unsigned region_base(mem_space_e sp, bit is_dst, bit hi);
    case (sp)
      SP_DDR:   return is_dst ? 64'h2000_0000 : (hi ? 64'h8_1000_0000 : 64'h1000_0000);
      SP_UB:    return tpu_pkg::UB_BASE;
      SP_ACC:   return tpu_pkg::ACC_BASE;
      default:  return tpu_pkg::WFIFO_BASE;
    endcase
  endfunction
  static function int unsigned region_size(mem_space_e sp);
    case (sp)
      SP_DDR:  return 32'h0100_0000;     // 16 MiB
      SP_UB:   return 32'h0001_0000;     // 64 KiB hot region of the UB
      SP_ACC:  return 32'h0000_8000;     // 32 KiB hot region of the ACC
      default: return 32'h0010_0000;     // the 1 MiB Weight FIFO window (address ignored)
    endcase
  endfunction

  constraint c_src   { src_sp != SP_WFIFO; }
  constraint c_len   { len dist { [1:15] :/ 10, 16 :/ 5, [17:255] :/ 20, [256:4095] :/ 40,
                                  4096 :/ 5, [4097:6000] :/ 20 }; }
  constraint c_reps  { reps dist { 1 :/ 65, [2:6] :/ 35 }; reps inside {[1:6]}; len inside {[1:6000]}; }
  constraint c_str   { sstride inside {[0:9000]}; dstride inside {[0:9000]};
                       (reps > 1) -> (sstride >= len && dstride >= len); }
  // FIFO windows need FULL beats: with a misaligned source iDMA's legalizer cuts the transfer at
  // the source 4 KiB page, which splits one destination word into two partial beats; the
  // window rejects them (SLVERR + flag) and the word is lost (UVM finding, L36). Rule: source
  // must be 16 B aligned too - Phase B READ_WEIGHTS encodes addr[48:4], so it always is.
  constraint c_fifo  { (dst_sp == SP_WFIFO) -> (dst_off == 0 && reps == 1 && len % 16 == 0 && len > 0 &&
                                                src_off % 16 == 0); }
  // 64-bit arithmetic: in 32-bit unsigned the solver satisfied this by WRAPPING (offsets like
  // 0xfffffd90 -> unmapped addresses; first UVM run)
  constraint c_fit   { src_off < region_size(src_sp); dst_off < region_size(dst_sp);
                       longint'(src_off) + longint'(reps - 1) * longint'(sstride) + longint'(len) <= longint'(region_size(src_sp));
                       longint'(dst_off) + longint'(reps - 1) * longint'(dstride) + longint'(len) <= longint'(region_size(dst_sp)); }
  // same on-chip memory as source and destination deadlocks (UVM finding, see tpu_same_mem_test);
  // excluded from random traffic - the runtime never issues such a copy
  constraint c_nosame { (src_sp inside {SP_UB, SP_ACC}) -> (dst_sp != src_sp); }
  constraint c_disj  { (src_sp == dst_sp) ->
                         ((longint'(src_off) + longint'(reps - 1) * longint'(sstride) + longint'(len) <= longint'(dst_off)) ||
                          (longint'(dst_off) + longint'(reps - 1) * longint'(dstride) + longint'(len) <= longint'(src_off))); }
  constraint c_order { solve src_sp, dst_sp before len; solve len, reps before sstride, dstride; }

  function longint unsigned src_addr(); return region_base(src_sp, 0, src_hi) + src_off; endfunction
  function longint unsigned dst_addr(); return region_base(dst_sp, 1, 0) + dst_off; endfunction
  function bit crosses_4k();
    return ((src_addr() & 12'hFFF) + len > 4096) || ((dst_addr() & 12'hFFF) + len > 4096);
  endfunction

  `uvm_object_utils_begin(dma_xfer)
    `uvm_field_enum(mem_space_e, src_sp, UVM_ALL_ON)
    `uvm_field_enum(mem_space_e, dst_sp, UVM_ALL_ON)
    `uvm_field_int(src_off, UVM_ALL_ON | UVM_HEX)
    `uvm_field_int(dst_off, UVM_ALL_ON | UVM_HEX)
    `uvm_field_int(len,     UVM_ALL_ON | UVM_DEC)
    `uvm_field_int(reps,    UVM_ALL_ON | UVM_DEC)
    `uvm_field_int(sstride, UVM_ALL_ON | UVM_DEC)
    `uvm_field_int(dstride, UVM_ALL_ON | UVM_DEC)
  `uvm_object_utils_end
  function new(string name = "dma_xfer"); super.new(name); endfunction

  function string brief();
    return $sformatf("%s+0x%0h -> %s+0x%0h len=%0d reps=%0d ss=%0d ds=%0d",
                     src_sp.name(), src_off, dst_sp.name(), dst_off, len, reps, sstride, dstride);
  endfunction
endclass
