# Test importing the preqtorch package
try:
    import preqtorch
    print(f"Successfully imported preqtorch version {preqtorch.__version__}")
    print("Available classes and functions:")
    for item in preqtorch.__all__:
        print(f"  - {item}")
except ImportError as e:
    print(f"Error importing preqtorch: {e}")
    print("Please make sure the package is installed with 'pip install -e .'")