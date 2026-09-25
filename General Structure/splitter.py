import sys
import os
import config

# (sura, ayah) marking the FIRST verse of each Juz, in order 1 -> 30.
JUZ_START_BOUNDARIES = [
    (1, 1),      # Juz 1
    (2, 142),    # Juz 2
    (2, 253),    # Juz 3
    (3, 93),     # Juz 4
    (4, 24),     # Juz 5
    (4, 148),    # Juz 6
    (5, 82),     # Juz 7
    (6, 111),    # Juz 8
    (7, 88),     # Juz 9
    (8, 41),     # Juz 10
    (9, 93),     # Juz 11
    (11, 6),     # Juz 12
    (12, 53),    # Juz 13
    (15, 1),     # Juz 14
    (17, 1),     # Juz 15
    (18, 75),    # Juz 16
    (21, 1),     # Juz 17
    (23, 1),     # Juz 18
    (25, 21),    # Juz 19
    (27, 56),    # Juz 20
    (29, 46),    # Juz 21
    (33, 31),    # Juz 22
    (36, 28),    # Juz 23
    (39, 32),    # Juz 24
    (41, 47),    # Juz 25
    (46, 1),     # Juz 26
    (51, 31),    # Juz 27
    (58, 1),     # Juz 28
    (67, 1),     # Juz 29
    (78, 1),     # Juz 30
]

def parse_tanzil_file(path):
    # Yields (sura, ayah, text) tuples in file order
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            sura, ayah, text = parts
            yield int(sura), int(ayah), text.strip()

def assign_juz(sura, ayah):
    # Returns which Juz (1-30) a given (sura, ayah) belongs to
    juz = 1
    for i, (start_sura, start_ayah) in enumerate(JUZ_START_BOUNDARIES):
        if (sura, ayah) >= (start_sura, start_ayah):
            juz = i + 1
        else:
            break
    return juz


def main(source_path):
    if not os.path.exists(source_path):
        print(f"File not found: {source_path}")
        sys.exit(1)

    os.makedirs(config.DOCS_DIR, exist_ok=True)

    juz_buffers = {n: [] for n in range(1, 31)}
    current_sura = None

    for sura, ayah, text in parse_tanzil_file(source_path):
        juz = assign_juz(sura, ayah)
        if sura != current_sura:
            juz_buffers[juz].append(f"\n=== Sura {sura} ===\n")
            current_sura = sura
        juz_buffers[juz].append(f"{sura}:{ayah} {text}")

    for n in range(1, 31):
        verses = juz_buffers[n]
        if not verses:
            print(f"  Warning: Juz {n} came out empty — check boundaries/source file.")
            continue
        out_path = os.path.join(config.DOCS_DIR, f"Para{n}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(verses))
        print(f"Wrote {out_path} ({len(verses)} verse lines)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_quran_by_juz.py path/to/quran-simple.txt")
        sys.exit(1)
    main(sys.argv[1])