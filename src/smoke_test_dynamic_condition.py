from src.train_dynamic import dynamic_update_condition


def main() -> None:
    assert dynamic_update_condition(5.0, 1.0, 2.0) is True
    assert dynamic_update_condition(1.0, 1.0, 2.0) is False
    assert dynamic_update_condition(-5.0, -1.0, -2.0) is True
    assert dynamic_update_condition(-1.0, -1.0, -2.0) is False
    assert dynamic_update_condition(10.0, 999.0, 0.0) is True
    print("Dynamic update-condition smoke test passed")


if __name__ == "__main__":
    main()
