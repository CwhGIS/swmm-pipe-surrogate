"""List the non-empty task compositions used by the clean demonstration."""

from .schema import all_task_sets


def main() -> None:
    task_sets = all_task_sets()
    print(f"Total non-empty task sets: {len(task_sets)}")
    for title, size in (("STL", 1), ("Selective", 2), ("Full", 5)):
        print(f"\n{title}:")
        for task_set in task_sets:
            if len(task_set) == size:
                print("+".join(task_set))


if __name__ == "__main__":
    main()
