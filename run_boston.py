from agenticml import ml

result = ml.run("BostonHousing.csv")
print("\n=== Pipeline Complete ===")
print(f"Status: {result.get('status')}")
print(f"Run dir: {result.get('run_dir')}")
