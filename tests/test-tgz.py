from pathlib import Path
import re
import sys
import tarfile


def print_green(msg: str):
    print(f"\033[32m{msg}\033[0m")


def print_cyan(msg: str):
    print(f"\033[96m{msg}\033[0m")


def print_red(msg: str):
    print(f"\033[31m{msg}\033[0m")


def validate_id(id_number: str, id_idx: int) -> bool:
    if not id_number.isdigit():
        print_red(f"ID {id_idx} is not made of digits")
        return False
    
    if len(id_number) > 9:
        print_red(f"ID {id_idx} is too long")
        return False
    elif len(id_number) < 9:
        print_red(f"ID {id_idx} is too short")
        return False
    
    # Calculate control digit
    control_digit = 0
    for i in range(8):
        d = int(id_number[i])
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        control_digit += d

    control_digit = 10 - (control_digit % 10)
    if control_digit == 10:
        control_digit = 0

    if control_digit != int(id_number[8]):
        print_red(f"ID {id_idx} is invalid")
        return False
    
    return True


def validate_tgz_filename(tgz_path: Path):
    success = True
    tgz_name = None
    if not tgz_path.exists():
        print_red("Error: no such file exists")
        success = False
    
    if len(tgz_path.suffixes) != 2 or tgz_path.suffixes[0] != ".tar" and tgz_path.suffixes[1] != ".gz":
        print_green("Error: file suffix should be .tar.gz")
        success = False
    
    tgz_name = tgz_path.name.removesuffix(".tar.gz")
    tgz_name_parts = tgz_name.split("_")
    tgz_name_error = False
    if len(tgz_name_parts) != 3:
        tgz_name_error = True
    else:
        if not validate_id(tgz_name_parts[0], 1):
            tgz_name_error = True
        
        if tgz_name_parts[1] != "111111111" and not validate_id(tgz_name_parts[1], 2):
            tgz_name_error = True
        
        if tgz_name_parts[2] != "project":
            tgz_name_error = True
    
    if tgz_name_error:
        print_red("Error: filename is invalid")
        success = False

    return success, tgz_name


def main():
    tgz_path = Path(sys.argv[1])
    tgz_name_valid, base_name = validate_tgz_filename(tgz_path)

    if base_name is None:
        exit(1)
    
    has_dir = False
    has_files = {
        "symnmf.py": False,
        "symnmf.c": False,
        "symnmfmodule.c": False,
        "symnmf.h": False,
        "analysis.py": False,
        "setup.py": False,
        "Makefile": False,
    }

    err = False
    
    with tarfile.open(tgz_path) as tar:
        for tarinfo in tar.getmembers():
            unknown_file = True
            if tarinfo.isdir() and tarinfo.name == base_name:
                has_dir = True
                unknown_file = False
                print_green(f"✅ {base_name} directory is found")
            elif tarinfo.isfile():
                for filename in has_files:
                    if tarinfo.name == f"{base_name}/{filename}":
                        has_files[filename] = True
                        unknown_file = False
                        print_green(f"✅ {filename} is found")
                        break
                else:
                    if match := re.match(
                        rf"{base_name}/(\w+\.(c|h|py))$", tarinfo.name
                    ):
                        unknown_file = False
                        print_cyan(f"💡 {match.group(1)} is found")

            if unknown_file:
                print_red(f"🚨 unknown file: {tarinfo.name}")
                err = True

    if not has_dir:
        print_red(f"❌ missing {base_name} directory")
        err = True
    else:
        for filename in filter(lambda x: not has_files[x], has_files):
            print_red(f"❌ missing {filename}")
            err = True
    
    if err:
        exit(1)


if __name__ == "__main__":
    main()