# TPU-KV260: Architecture Diagrams, Bottom to Top

Current state: **v1.3, Phase B + CNN** (`VERSION = 0x0000_0500`, ABI v1).
This page walks the design from the lowest level (one PE) up to the software stack. Each diagram is cross-checked against the RTL instance hierarchy. Details live in the documents linked at the end of each section. Register map and address map: [`interfaces.md`](interfaces.md). ISA: [`isa.md`](isa.md).

Contents:
1. [Layer 1: Processing element and systolic array](#1-processing-element-and-systolic-array)
2. [Layer 2: Matrix unit](#2-matrix-unit)
3. [Layer 3: Memories](#3-on-chip-memories)
4. [Layer 4: Activation and pooling](#4-activation-and-pooling-unit)
5. [Layer 5: Control (decoder and sequencers)](#5-control-decoder-and-sequencers)
6. [Layer 6: `tpu_core`](#6-tpu_core)
7. [Layer 7: Data fabric (`tpu_fabric`)](#7-data-fabric-tpu_fabric)
8. [Layer 8: iDMA subsystem](#8-idma-subsystem)
9. [Layer 9: `kv260_top` and the PS](#9-kv260_top-and-the-zynqmp-ps)
10. [Layer 10: Linux driver and runtime](#10-linux-driver-and-user-space-runtime)
11. [Layer 11: Model to inference flow](#11-from-model-to-inference)
12. [Run lifecycle (sequence)](#12-run-lifecycle)

---

## 1. Processing element and systolic array

`rtl/compute/mac_pe_ws.sv`, `rtl/compute/systolic_array.sv`

A **weight-stationary** 16×16 array. Each PE holds one INT8 weight (with a shadow register for double buffering), multiplies it by the activation passing left to right, and adds the partial sum arriving from above. The multiplier maps to a DSP48E2 (`use_dsp`).

```mermaid
flowchart TB
    subgraph PE["mac_pe_ws (one PE)"]
        direction LR
        WS["weight shadow reg<br/>(INT8)"] -->|"load pulse"| WA["active weight reg<br/>(INT8)"]
        AIN["activation in (INT8)<br/>from the left"] --> MUL["INT8 x INT8<br/>DSP48E2 multiply"]
        WA --> MUL
        MUL --> ADD["+"]
        PIN["partial sum in (INT32)<br/>from above"] --> ADD
        ADD --> PREG["psum reg (INT32)"]
        AIN --> AREG["activation reg"]
    end
    PREG -->|"down"| PS_OUT["psum out to the PE below"]
    AREG -->|"right"| A_OUT["activation to the PE on the right"]
```

```mermaid
flowchart LR
    subgraph ARR["systolic_array (N = 16)"]
        direction TB
        R0["row 0: PE00 -> PE01 -> ... -> PE0,15"]
        R1["row 1: PE10 -> PE11 -> ... -> PE1,15"]
        RD["..."]
        R15["row 15: PE15,0 -> ... -> PE15,15"]
        R0 -. "psum flows down" .-> R1 -.-> RD -.-> R15
    end
    DS["data_setup<br/>skews activations<br/>(row i delayed i cycles)"] -->|"16 x INT8 per cycle"| ARR
    WL["weight_loader<br/>weight tile shift-in<br/>16 cycles per tile"] -->|"top edge, shadow regs"| ARR
    ARR -->|"16 x INT32 psums<br/>from the bottom edge"| OD["output_deskew<br/>aligns the diagonal wavefront"]
```

Weight tile word `t` = `W_tile[15-t]` (the loader shifts the last row in first). One matrix instruction: stream N input rows through the array and receive N rows of 16 INT32 sums, 16 cycles of latency after the last input.

---

## 2. Matrix unit

`rtl/compute/matrix_unit.sv`

```mermaid
flowchart LR
    WF[("Weight FIFO<br/>1024 words")] -->|"valid/ready, 16 B words"| WL["weight_loader<br/>i_wload"]
    UBR["UB read data<br/>(16 x INT8)"] --> DS["data_setup<br/>i_setup"]
    WL --> SA["systolic_array<br/>i_array"]
    DS --> SA
    SA --> OD["output_deskew<br/>i_deskew"]
    OD -->|"16 x INT32 row"| ACCW["to Accumulators<br/>(write or accumulate)"]
    CTRL["matmul_seq<br/>commands"] --> WL
    CTRL --> DS
```

---

## 3. On-chip memories

`rtl/memory/*`. All are accessed from two sides: **port A = AXI** (iDMA and CPU, through `axi_to_detailed_mem`) and **port B = TPU-native** (wide port, no arbitration).

```mermaid
flowchart TB
    subgraph UB["unified_buffer (1 MiB)"]
        UBA["axi_to_detailed_mem<br/>(port A, AXI side)"] --> UBR[["ram_tdp<br/>65,536 rows x 16 B<br/>URAM via XPM<br/>read latency 3"]]
        UBN["native port B<br/>ub_en, ub_we[15:0], ub_addr[15:0]"] --> UBR
    end
    subgraph ACC["accumulators (128 KiB)"]
        ACA["axi_to_detailed_mem<br/>(port A)"] --> ACR[["4 x ram_tdp banks<br/>2048 rows x 16 x INT32 (64 B)<br/>read latency 1"]]
        ACN["native port B<br/>whole 512-bit row, acc_we[3:0]"] --> ACR
    end
    subgraph WFIFO["stream_window: Weight FIFO"]
        WFA["axi_to_detailed_mem<br/>push on full-strobe write"] --> WFR[["fifo_ram<br/>1024 x 16 B"]]
        WFR --> WFN["first-word-fall-through<br/>to weight_loader"]
    end
    subgraph IFIFO["stream_window: Instruction FIFO"]
        IFA["axi_to_detailed_mem<br/>push on full-strobe write"] --> IFR[["fifo_ram<br/>2048 x 16 B"]]
        IFR --> IFN["to decoder"]
    end
```

| Memory | Size | Shape | AXI window |
|---|---|---|---|
| Unified Buffer | 1 MiB | 65,536 rows x 16 x INT8 | `0xA010_0000` |
| Accumulators | 128 KiB | 2,048 rows x 16 x INT32 | `0xA020_0000` |
| Weight FIFO | 16 KiB | 1,024 words (64 tiles) | `0xA030_0000` |
| Instruction FIFO | 32 KiB | 2,048 x 16 B instructions | `0xA040_0000` |

Ownership: while `CTRL[1]` (UB) or `CTRL[2]` (ACC) is set, CPU and register-frontend writes are dropped with SLVERR. Core-issued DMA (AXI user bit 0 = 1) is allowed (ADR-0003).

---

## 4. Activation and pooling unit

`rtl/activation/*`

```mermaid
flowchart LR
    ACCR["Accumulator row<br/>16 x INT32"] --> BIAS["+ bias<br/>(zero point folded in)"]
    BIAS --> RQ["requant_lane x 16<br/>multiplier + shift<br/>TFLite rounding"]
    PAR["act_params<br/>per-channel multiplier,<br/>shift, bias"] --> RQ
    RQ --> CL["clamp<br/>ReLU / ReLU6 / int8"]
    CL --> POOL{"pool field"}
    POOL -->|"0: none"| OUT["INT8 row to UB"]
    POOL -->|"1: 2x2 max<br/>2: 2x2 avg"| PU["pool_unit<br/>(after requantization)"]
    PU --> OUT
```

---

## 5. Control: decoder and sequencers

`rtl/control/*`

```mermaid
flowchart TB
    IF[("Instruction FIFO<br/>16 B words")] --> DEC["decoder"]
    DEC -->|"READ_WEIGHTS /<br/>READ_HOST_MEM /<br/>WRITE_HOST_MEM"| ARB["idma_request_arbiter<br/>(core request, source-tagged)"]
    DEC -->|"MATRIX_MULTIPLY /<br/>CONVOLVE"| MM["matmul_seq"]
    DEC -->|"ACTIVATE"| AS["act_seq"]
    DEC -->|"SET_CONFIG"| CFG["config registers<br/>(0x4000-0x4003 geometry,<br/>cleared on RUN edge)"]
    DEC -->|"INTERRUPT_HOST / HALT"| IRQ["irq_pending, halted"]
    DEC -.->|"sync: RAW hazard<br/>wait for sequencers"| MM
    DEC -.-> AS
    CFG --> CAG
    MM --> CAG["conv_address_gen<br/>implicit im2col:<br/>one kernel tap per instruction,<br/>padding supplies input zero point"]
    CAG --> UBRD["UB read address stream"]
    MM -->|"matrix commands"| MXU["matrix_unit"]
    AS -->|"ACC read, then UB write"| AU["activation_unit + pool_unit"]
    PERF["perf_counters<br/>total, array_active, weight_stall,<br/>weight_shift, non_matrix, raw_stall,<br/>act_active, instr_count"]
    DEC --> PERF
    MM --> PERF
    AS --> PERF
```

Instruction set (16 B, [`isa.md`](isa.md)):

| Group | Instructions |
|---|---|
| DMA | READ_HOST_MEM, READ_WEIGHTS, WRITE_HOST_MEM |
| Compute | MATRIX_MULTIPLY, CONVOLVE, ACTIVATE |
| Control | SET_CONFIG, SYNC, INTERRUPT_HOST, NOP, HALT |

---

## 6. `tpu_core`

`rtl/top/tpu_core.sv`: instruction FIFO consumer plus the full compute pipeline.

```mermaid
flowchart LR
    IFIFO[("Instruction FIFO")] --> DEC["decoder<br/>i_dec"]
    DEC --> MM["matmul_seq<br/>i_mm"]
    DEC --> AS["act_seq<br/>i_act"]
    DEC --> PERF["perf_counters<br/>i_perf"]
    DEC ==>|"core DMA request"| DMAREQ(["to idma_request_arbiter"])
    WFIFO[("Weight FIFO")] --> MXU
    MM --> MXU["matrix_unit<br/>i_mxu"]
    UB[("Unified Buffer<br/>native port")] <--> MM
    MXU --> ACC[("Accumulators<br/>native port")]
    ACC --> AS
    AS --> AU["activation_unit<br/>i_au (+ pool_unit)"]
    AU --> UB
    DEC --> IRQ(["irq, halted, cfg_err,<br/>debug tag, dma_fault"])
```

---

## 7. Data fabric (`tpu_fabric`)

`rtl/top/tpu_fabric.sv`: a PULP `axi_xbar` with an `axi_cut` on every slave, which fixed the timing paths into the RAMs (L33).

```mermaid
flowchart LR
    M0["master 0: iDMA<br/>(core user bit = 1<br/>for core DMA)"] --> XB
    M1["master 1: CPU HPM0<br/>(debug, user bit = 0)"] --> XB
    subgraph XB["axi_xbar (128-bit)"]
        DEC2["address decode<br/>any other address: DECERR (never hangs, L5)"]
    end
    XB --> C0["axi_cut"] --> UBW["unified_buffer"]
    XB --> C1["axi_cut"] --> ACW["accumulators"]
    XB --> C2["axi_cut"] --> WFW["stream_window: Weight FIFO"]
    XB --> C3["axi_cut"] --> IFW["stream_window: Instruction FIFO"]
    XB --> C4["axi_cut"] --> HP0["HP0 to PS DDR<br/>(iDMA only; HPM0 to DDR is masked)"]
```

---

## 8. iDMA subsystem

`rtl/top/idma_subsystem.sv`: PULP iDMA v0.7.4. This replaced the AMD AXI DMA IP.

```mermaid
flowchart LR
    CFG["AXI4-Lite config<br/>0xA000_1000"] --> A2P["axi_lite_to_apb"]
    A2P --> FE["idma_reg64_2d<br/>64-bit src/dst + 2D strides<br/>(read next_id launches)"]
    FE --> ARB["idma_request_arbiter"]
    CORE["core DMA request<br/>from decoder"] --> ARB
    ARB --> ND["idma_nd_midend<br/>NumDim = 2"]
    ND --> BE["idma_backend_rw_axi<br/>DataWidth 128, AddrWidth 49<br/>hardware legalizer: 4 KiB / 256-beat"]
    BE --> JOIN["axi_rw_join"]
    JOIN --> AXIM["AXI master to tpu_fabric xbar"]
    ND -->|"response"| IDG["idma_transfer_id_gen<br/>next_id, done_id"]
    IDG --> FE
    IDG --> DONE["IDMA_DONE_ID reg"]
    FE -. "busy / error" .-> STATUS["STATUS register"]
```

Rules enforced by hardware and the driver: FIFO-window DMA needs a 16 B-aligned source and destination (L36). No same-memory DMA (UB to UB, ACC to ACC) because it deadlocks (L34).

---

## 9. `kv260_top` and the ZynqMP PS

`rtl/top/kv260_top.sv`: one clock domain (`pl_clk0`, 100 MHz), no CDC.

```mermaid
flowchart TB
    subgraph PS["zynq_ultra_ps_e (Cortex-A53 + Linux)"]
        HPM0["HPM0_FPD (AXI4, 128-bit)<br/>0xA000_0000 + 8 MiB"]
        HP0["HP0_FPD slave (128-bit)"]
        DDR[("PS DDR<br/>0..2 GiB and 0x8_0000_0000..")]
        GIC["GIC SPI 121<br/>(pl_ps_irq0[0])"]
        HP0 --- DDR
    end
    subgraph PL["kv260_top (PL)"]
        IW["axi_iw_converter"] --> DX["axi_demux_simple<br/>addr < 0xA010_0000 : config path<br/>otherwise: data path"]
        DX -->|"config"| DW["axi_dw_converter<br/>128 to 32"]
        DW --> AL["axi_to_axi_lite"]
        AL --> LX["axi_lite_xbar"]
        LX --> REGS["tpu_regs (axi_lite_regs)<br/>0xA000_0000"]
        LX --> IDMA["idma_subsystem<br/>0xA000_1000"]
        DX -->|"data"| FAB["tpu_fabric<br/>(UB, ACC, WFIFO, IFIFO, xbar)"]
        IDMA -->|"AXI master"| FAB
        FAB --> HP0P["to HP0"]
        FAB <--> CORE["tpu_core<br/>(decoder, matrix unit, activation)"]
        REGS -->|"CTRL: run, owner, flush,<br/>irq clear, perf clear"| CORE
        CORE -->|"CORE_STATUS, PERF0-7,<br/>irq level"| REGS
    end
    HPM0 --> IW
    HP0P --> HP0
    REGS --> GIC
```

Register summary (`0xA000_0000`): VERSION, SCRATCH, BUILD_ID, IDMA_DONE_ID, STATUS, CTRL, FIFO_LEVEL, CORE_STATUS, IRQ_TAG, PERF0-7, DEBUG_TAG ([`interfaces.md`](interfaces.md) section 2).

---

## 10. Linux driver and user-space runtime

`software/driver`, `software/runtime`

```mermaid
flowchart TB
    subgraph USER["User space"]
        APP["application / model bundle"] --> RT["libtpu (C) / tpu.py<br/>tpu_mlp.py, run_case.py"]
        COMP["compilers<br/>compile_mlp.py, compile_cnn.py,<br/>compile_phase_b.py"] -->|"prog.bin + relocations +<br/>weights + input"| RT
    end
    subgraph KERNEL["Kernel: tpu_kv260.ko"]
        MISC["/dev/tpu0 (misc device)<br/>ioctls: GET_INFO, ALLOC, FREE, RUN, RECOVER"]
        VAL["validation<br/>relocation bounds, DMA-op only,<br/>16 B alignment, weight-underflow static check"]
        BUF["DMA-coherent buffers<br/>(no IOMMU: kernel relocates DMA addresses)"]
        STM["state machine<br/>IDLE, RUNNING, WEDGED<br/>drain/recover never resets the PL (L30)"]
        ISR["IRQ handler (GIC 121)<br/>latches perf counters, clears CTRL[8]"]
    end
    subgraph HW["Hardware"]
        FAB2["tpu_regs + fabric<br/>a0000000.fabric"]
        MEM[("PS DDR<br/>input, weights, program, output")]
    end
    RT -->|"ioctl / mmap"| MISC
    MISC --> VAL --> BUF
    MISC --> STM
    STM -->|"MMIO: copy program to IFIFO via iDMA,<br/>CTRL run"| FAB2
    FAB2 -->|"iDMA / core DMA"| MEM
    FAB2 -->|"interrupt"| ISR --> STM
    BUF --- MEM
```

The driver binds to `a0000000.fabric` through `driver_override`. Load with `software/driver/tpu-load.sh`; unload with `tpu-unload.sh`. See [`driver.md`](driver.md) and [`driver_user_guide.md`](driver_user_guide.md).

---

## 11. From model to inference

```mermaid
flowchart LR
    TR["train / export<br/>model/mlp/train_demo.py<br/>export_bundle.py"] --> Q["TFLite-style int8 quantization<br/>per-channel weights,<br/>zero point folded into bias,<br/>multiplier + shift"]
    Q --> CMP["compiler<br/>tile to 16x16,<br/>allocate UB / ACC,<br/>emit 16 B instructions<br/>(whole program, 2048 max)"]
    CMP --> GOLD["golden model<br/>model/isa/sim.py<br/>conv_ref.py"]
    CMP --> BUNDLE["bundle<br/>prog.bin, relocations.json,<br/>wmem, input"]
    BUNDLE --> DRV["/dev/tpu0 RUN"]
    DRV --> HWEX["hardware executes<br/>Phase B autonomous"]
    HWEX --> CHK{"output == golden?<br/>(bit-exact)"}
    GOLD --> CHK
```

---

## 12. Run lifecycle

One `RUN` ioctl, Phase B (the whole program is preloaded; the core issues its own DMA).

```mermaid
sequenceDiagram
    autonumber
    participant App as App / libtpu
    participant Drv as tpu_kv260.ko
    participant Reg as tpu_regs + iDMA
    participant Core as tpu_core
    participant DDR as PS DDR

    App->>Drv: ioctl(RUN, program, relocations)
    Drv->>Drv: validate and relocate DMA addresses
    Drv->>Reg: flush FIFOs, clear errors, perf clear
    Drv->>Reg: iDMA: program DDR to Instruction FIFO
    Drv->>Reg: CTRL: UB/ACC owner = TPU, run 0 to 1
    loop each instruction
        Core->>Core: decode (SET_CONFIG, SYNC, ...)
        Core->>DDR: READ_HOST_MEM / READ_WEIGHTS via iDMA (user bit = 1)
        DDR-->>Core: input rows to UB, weight tiles to Weight FIFO
        Core->>Core: MATRIX_MULTIPLY / CONVOLVE (array, accumulators)
        Core->>Core: ACTIVATE (requantize, clamp, pool) to UB
        Core->>DDR: WRITE_HOST_MEM via iDMA
    end
    Core->>Reg: INTERRUPT_HOST, HALT (irq level, GIC 121)
    Reg-->>Drv: interrupt
    Drv->>Reg: latch perf counters, CTRL[8] irq clear
    Drv-->>App: status, perf (output already in DDR)
```

---

## Related documents

| Topic | File |
|---|---|
| Full spec and milestones | [`architecture.md`](architecture.md) |
| Address map, registers, memory windows | [`interfaces.md`](interfaces.md) |
| Instruction encoding | [`isa.md`](isa.md) |
| iDMA parameters and programming | [`idma_integration.md`](idma_integration.md) |
| Core DMA contract | [`m5_core_contract.md`](m5_core_contract.md), [`m5_dma_interface.md`](m5_dma_interface.md) |
| CNN (CONVOLVE, pooling) | [`m6_cnn_contract.md`](m6_cnn_contract.md), [`adr/0004-cnn-convolve-pool.md`](adr/0004-cnn-convolve-pool.md) |
| Driver | [`driver.md`](driver.md), [`driver_user_guide.md`](driver_user_guide.md) |
