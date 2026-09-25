"""Build the molecule_roles index: python build_index.py [path/to/molecules.sqlite]"""
import sqlite3, sys, time

db = sys.argv[1] if len(sys.argv) > 1 else "molecules.sqlite"
con = sqlite3.connect(db)

if con.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='molecule_roles'").fetchone():
    print("molecule_roles already present; dropping and rebuilding")
    con.execute("DROP TABLE molecule_roles")

t0 = time.time()
con.execute("CREATE TABLE molecule_roles (bit INTEGER NOT NULL, mol_id INTEGER NOT NULL,"
            " PRIMARY KEY (bit, mol_id)) WITHOUT ROWID")


def rows():
    for mol_id, blob in con.execute("SELECT mol_id, role_mask FROM molecules"):
        mask = int.from_bytes(bytes(blob), "big")
        bit = 0
        while mask:
            if mask & 1:
                yield bit, mol_id
            mask >>= 1
            bit += 1


con.executemany("INSERT INTO molecule_roles VALUES (?,?)", rows())
con.commit()
n = con.execute("SELECT COUNT(*) FROM molecule_roles").fetchone()[0]
print(f"{n:,} rows in {time.time() - t0:.1f}s")
