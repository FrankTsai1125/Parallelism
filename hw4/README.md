# HW4: Bitcoin Miner

## Files
- `hw4.cu`: Main implementation
- `sha256.cu`, `sha256.h`: SHA-256 hash functions
- `Makefile`: Build configuration
- `REPORT.txt`: Detailed report

## Build
```bash
make clean
make
```

## Run
```bash
./hw4 <input_file> <output_file>
```

## Implementation Highlights
1. **Persistent Kernel**: Work-stealing queue for single-block mining
2. **Static Stream Pool**: Reusable CUDA resources for multi-block
3. **Zero-Copy Memory**: Pinned memory eliminates synchronization
4. **OpenMP Preprocessing**: Parallel midstate computation

## Performance
- Case00: ~2.0s
- Case01: ~3.7s
- Case02: ~0.7s
- Case03: ~0.4s

All test cases: **PASS** ✓

