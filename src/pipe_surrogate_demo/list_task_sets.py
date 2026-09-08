"""List the non-empty task compositions used by the clean demonstration."""

from .schema import all_task_sets


def group_task_sets(
    task_sets: tuple[tuple[str, ...], ...] | None = None,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Group all task subsets into STL, selective, and Full-MTL sections."""
    task_sets = task_sets or all_task_sets()
    return {
        "STL": tuple(task_set for task_set in task_sets if len(task_set) == 1),
        "Selective": tuple(
            task_set for task_set in task_sets if 2 <= len(task_set) <= 4
        ),
        "Full": tuple(task_set for task_set in task_sets if len(task_set) == 5),
    }


def main() -> None:
    groups = group_task_sets()
    task_count = sum(len(group) for group in groups.values())
    print(f"Total non-empty task sets: {task_count}")
    for title in ("STL", "Selective", "Full"):
        print(f"\n{title}:")
        for task_set in groups[title]:
            print("+".join(task_set))


if __name__ == "__main__":
    main()
