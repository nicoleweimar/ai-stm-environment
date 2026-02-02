def main() -> int:
    print("--AI-STM Check--")

    print("\n[1/2] Environment verification")
    from verify.verify_cuda import main as verify_main
    rc1 = verify_main()

    print("\n[2/2] CNN pipeline validation")
    from assessment.scripts.validate_cnn_pipeline import main as validate_main
    rc2 = validate_main()

    rc = 0 if (rc1 == 0 and rc2 == 0) else 1
    print("\n Check complete" if rc == 0 else "\n Check failed")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())