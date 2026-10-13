from pathlib import Path

import pandas as pd


def find_project_root(start=None):
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find the project root containing the 'data' folder.")


def list_data_dir(data_dir):
    print(f"Data folder: {data_dir}")
    for item in sorted(data_dir.iterdir()):
        kind = "directory" if item.is_dir() else "file"
        print(f"- {item.name} [{kind}]")
    print()


def main():
    project_root = find_project_root()
    data_dir = project_root / "data"
    list_data_dir(data_dir)

    excel_path = data_dir / "MetadatabyAnnotation.xlsx"
    if excel_path.exists():
        df = pd.read_excel(excel_path)
        print(f"Loaded metadata file: {excel_path.name}")
        print(df.head())
        print("\nColumns:", list(df.columns))
        print("Shape:", df.shape)
    else:
        print(f"Missing metadata file: {excel_path}")


if __name__ == "__main__":
    main()
