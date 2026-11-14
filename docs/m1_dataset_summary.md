# M1 Dataset Summary — NASA C-MAPSS

Source directory: `data/raw/cmapps/CMAPSSData/`

## Files

| File | Lines | Description |
|---|---|---|
| `train_FD001.txt` | 20,631 | Training set, subdataset FD001 |
| `test_FD001.txt` | 13,096 | Test set, subdataset FD001 |
| `RUL_FD001.txt` | 100 | Ground-truth RUL for FD001 test engines |
| `train_FD002.txt` | 53,759 | Training set, subdataset FD002 |
| `test_FD002.txt` | 33,991 | Test set, subdataset FD002 |
| `RUL_FD002.txt` | 259 | Ground-truth RUL for FD002 test engines |
| `train_FD003.txt` | 24,720 | Training set, subdataset FD003 |
| `test_FD003.txt` | 16,596 | Test set, subdataset FD003 |
| `RUL_FD003.txt` | 100 | Ground-truth RUL for FD003 test engines |
| `train_FD004.txt` | 61,249 | Training set, subdataset FD004 |
| `test_FD004.txt` | 41,214 | Test set, subdataset FD004 |
| `RUL_FD004.txt` | 248 | Ground-truth RUL for FD004 test engines |
| `readme.txt` | 45 | Dataset readme / format description |
| `Damage Propagation Modeling.pdf` | 1,775 | Reference paper on damage propagation |

Total: 267,783 lines across the four subdatasets (FD001–FD004).

## Notes

- Each subdataset has a train/test pair plus a corresponding `RUL_FDxxx.txt` file holding the true remaining-useful-life values for the test engines.
- Per project rules, splits must be made by `engine_id` (not random rows) to prevent data leakage.
- Files in `data/raw/` are read-only; never modify them.