// True dual-port RAM, single clock, byte write enables, NO-CHANGE (a write cycle does not
// update the read output; URAM supports only this mode), Latency 1 or 2 (2 adds an
// output register, needed for URAM cascades). Written to the UG901 inference templates so
// Vivado maps it to BRAM ("block") or URAM ("ultra") - never to flip-flops (L19).
// The ram_style attribute is a literal in each generate branch (tools may ignore a
// parameter-valued attribute).

module ram_tdp #(
  parameter int unsigned Depth    = 1024,
  parameter int unsigned Width    = 128,              // multiple of 8
  parameter int unsigned Latency  = 1,                // 1 or 2
  parameter string       RamStyle = "block",          // "block" | "ultra"
  localparam int unsigned AW = $clog2(Depth),
  localparam int unsigned BW = Width / 8
) (
  input  logic             clk_i,
  // port A
  input  logic             a_en_i,
  input  logic [BW-1:0]    a_we_i,
  input  logic [AW-1:0]    a_addr_i,
  input  logic [Width-1:0] a_wdata_i,
  output logic [Width-1:0] a_rdata_o,
  // port B
  input  logic             b_en_i,
  input  logic [BW-1:0]    b_we_i,
  input  logic [AW-1:0]    b_addr_i,
  input  logic [Width-1:0] b_wdata_i,
  output logic [Width-1:0] b_rdata_o
);

  logic [Width-1:0] a_q, b_q;

  if (RamStyle == "ultra") begin : gen_uram
`ifdef TPU_USE_XPM
    // Vivado ignores ram_style="ultra" on this TDP+byte-enable template (it silently built the
    // 1 MiB UB from 256 BRAM36 in M2), so URAM is instantiated through AMD's XPM macro.
    // Simulation (no TPU_USE_XPM) uses the behavioural model below, same latency/semantics.
    xpm_memory_tdpram #(
      .MEMORY_SIZE         ( Depth * Width ),
      .MEMORY_PRIMITIVE    ( "ultra"       ),
      .CLOCKING_MODE       ( "common_clock"),
      .MEMORY_INIT_FILE    ( "none"        ),
      .USE_MEM_INIT        ( 0             ),
      .WAKEUP_TIME         ( "disable_sleep"),
      .ECC_MODE            ( "no_ecc"      ),
      .AUTO_SLEEP_TIME     ( 0             ),
      .MESSAGE_CONTROL     ( 0             ),
      .WRITE_DATA_WIDTH_A  ( Width ), .READ_DATA_WIDTH_A ( Width ), .BYTE_WRITE_WIDTH_A ( 8 ),
      .ADDR_WIDTH_A        ( AW    ), .READ_LATENCY_A    ( Latency ), .WRITE_MODE_A ( "no_change" ),
      .READ_RESET_VALUE_A  ( "0"   ),
      .WRITE_DATA_WIDTH_B  ( Width ), .READ_DATA_WIDTH_B ( Width ), .BYTE_WRITE_WIDTH_B ( 8 ),
      .ADDR_WIDTH_B        ( AW    ), .READ_LATENCY_B    ( Latency ), .WRITE_MODE_B ( "no_change" ),
      .READ_RESET_VALUE_B  ( "0"   )
    ) i_xpm (
      .clka ( clk_i ), .rsta ( 1'b0 ), .ena ( a_en_i ), .regcea ( 1'b1 ),
      .wea  ( a_we_i ), .addra ( a_addr_i ), .dina ( a_wdata_i ), .douta ( a_rdata_o ),
      .clkb ( clk_i ), .rstb ( 1'b0 ), .enb ( b_en_i ), .regceb ( 1'b1 ),
      .web  ( b_we_i ), .addrb ( b_addr_i ), .dinb ( b_wdata_i ), .doutb ( b_rdata_o ),
      .sleep ( 1'b0 ),
      .injectsbiterra ( 1'b0 ), .injectdbiterra ( 1'b0 ),
      .injectsbiterrb ( 1'b0 ), .injectdbiterrb ( 1'b0 ),
      .sbiterra (), .dbiterra (), .sbiterrb (), .dbiterrb ()
    );
`else
    logic [Width-1:0] mem [Depth];
    always_ff @(posedge clk_i) begin
      if (a_en_i) begin
        if (|a_we_i) begin
          for (int i = 0; i < BW; i++) if (a_we_i[i]) mem[a_addr_i][i*8 +: 8] <= a_wdata_i[i*8 +: 8];
        end else a_q <= mem[a_addr_i];
      end
    end
    always_ff @(posedge clk_i) begin
      if (b_en_i) begin
        if (|b_we_i) begin
          for (int i = 0; i < BW; i++) if (b_we_i[i]) mem[b_addr_i][i*8 +: 8] <= b_wdata_i[i*8 +: 8];
        end else b_q <= mem[b_addr_i];
      end
    end
`endif
  end else begin : gen_bram
    (* ram_style = "block" *) logic [Width-1:0] mem [Depth];
    always_ff @(posedge clk_i) begin
      if (a_en_i) begin
        if (|a_we_i) begin
          for (int i = 0; i < BW; i++) if (a_we_i[i]) mem[a_addr_i][i*8 +: 8] <= a_wdata_i[i*8 +: 8];
        end else a_q <= mem[a_addr_i];
      end
    end
    always_ff @(posedge clk_i) begin
      if (b_en_i) begin
        if (|b_we_i) begin
          for (int i = 0; i < BW; i++) if (b_we_i[i]) mem[b_addr_i][i*8 +: 8] <= b_wdata_i[i*8 +: 8];
        end else b_q <= mem[b_addr_i];
      end
    end
  end

`ifdef TPU_USE_XPM
  localparam bit XpmOut = (RamStyle == "ultra");
`else
  localparam bit XpmOut = 1'b0;
`endif
  if (XpmOut) begin : gen_xpm_out
    // outputs driven by xpm_memory_tdpram
  end else if (Latency == 2) begin : gen_pipe
    logic [Width-1:0] a_q2, b_q2;
    always_ff @(posedge clk_i) begin
      a_q2 <= a_q;
      b_q2 <= b_q;
    end
    assign a_rdata_o = a_q2;
    assign b_rdata_o = b_q2;
  end else begin : gen_nopipe
    assign a_rdata_o = a_q;
    assign b_rdata_o = b_q;
  end

  // pragma translate_off
  initial begin
    assert (Width % 8 == 0)               else $fatal(1, "ram_tdp: Width must be a multiple of 8");
    assert (Latency == 1 || Latency == 2) else $fatal(1, "ram_tdp: Latency must be 1 or 2");
  end
  // pragma translate_on

endmodule
