// UVM register model of the iDMA reg64_2d frontend (offsets: docs/idma_integration.md).
// 64-bit registers are accessed as two 32-bit bus beats, low word first (UVM little endian),
// which is exactly the order the frontend expects. next_id / done_id / status are volatile RO:
// reading next_id LAUNCHES a transfer, so only the dma sequences read it, exactly once.

class tpu_reg extends uvm_reg;
  uvm_reg_field val;
  function new(string name = "tpu_reg", int unsigned n_bits = 32);
    super.new(name, n_bits, UVM_NO_COVERAGE);
  endfunction
  function void build_field(string access, bit volatile);
    val = uvm_reg_field::type_id::create("val");
    val.configure(this, get_n_bits(), 0, access, volatile, 0, 1, 0, 1);
  endfunction
endclass

class idma_reg_block extends uvm_reg_block;
  `uvm_object_utils(idma_reg_block)
  tpu_reg conf, status0, next_id, done_id;
  tpu_reg dst_addr, src_addr, length, dst_stride, src_stride, reps;

  function new(string name = "idma_reg_block"); super.new(name, UVM_NO_COVERAGE); endfunction

  local function tpu_reg mk(string n, int unsigned bits, string acc, bit vol, uvm_reg_addr_t off);
    tpu_reg r = new(n, bits);
    r.configure(this);
    r.build_field(acc, vol);
    default_map.add_reg(r, off, (acc == "RO") ? "RO" : "RW");
    return r;
  endfunction

  virtual function void build();
    default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN, 1);
    conf       = mk("conf",       32, "RW", 0, 'h000);
    status0    = mk("status0",    32, "RO", 1, 'h004);
    next_id    = mk("next_id",    32, "RO", 1, 'h044);
    done_id    = mk("done_id",    32, "RO", 1, 'h084);
    dst_addr   = mk("dst_addr",   64, "RW", 0, 'h0D0);
    src_addr   = mk("src_addr",   64, "RW", 0, 'h0D8);
    length     = mk("length",     64, "RW", 0, 'h0E0);
    dst_stride = mk("dst_stride", 64, "RW", 0, 'h100);
    src_stride = mk("src_stride", 64, "RW", 0, 'h108);
    reps       = mk("reps",       64, "RW", 0, 'h110);
    lock_model();
  endfunction
endclass
