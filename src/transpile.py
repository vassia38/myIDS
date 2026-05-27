import joblib
import m2cgen as m2c

# 1. Load your trained model
print("Loading model...")
rfc = joblib.load("best_lightweight_multiclass_rfc.sav")

# 2. Transpile the Python model directly into Go code
print("Transpiling to Go...")
go_code = m2c.export_to_go(rfc)

# 3. Save it to a Go file
with open("rf_model.go", "w") as f:
    f.write("package main\n\n")
    f.write(go_code)

print("Successfully exported to rf_model.go!")
