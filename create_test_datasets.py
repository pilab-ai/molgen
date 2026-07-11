#!/usr/bin/env python3
"""Generate synthetic test datasets for molgen.

Creates minimal but valid PyG InMemoryDataset .pt files for QM9, PI
(repeat unit + full + latent), and SDBS, so the training pipelines can
be tested end-to-end without real data.

Usage:
    python3 create_test_datasets.py [--root DIR]

Output directory tree (under <root>):
    dataset/qm9/data/processed/data_v3.pt
    dataset/pi/data/processed/data_repeat_unit.pt
    dataset/pi/data/processed/data.pt
    dataset/pi/data/processed/data_with_latent.pt
    dataset/pi/data/processed/code_stats.pt
    dataset/sdbs/data/processed/data.pt
"""

import argparse
import os
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collate(data_list):
    """Minimal re-implementation of torch_geometric.data.collate.

    Builds the (data, slices) tuple that InMemoryDataset.__init__ loads.
    Each attribute is concatenated along dim=0 and a slices tensor records
    the per-graph boundaries.
    """
    keys = list(data_list[0].keys())
    data_dict = {}
    slices_dict = {}

    for key in keys:
        cat_dim = 0  # default cat_dim for all attributes
        # edge_index is [2, E], concatenate along dim=1
        if key == "edge_index" or key == "edge_attr":
            cat_dim = 1 if key == "edge_index" else 0

        values = []
        for d in data_list:
            v = d[key]
            if isinstance(v, torch.Tensor):
                values.append(v)
            else:
                # scalar → tensor
                values.append(torch.tensor([v]))

        if key == "edge_index":
            # edge_index shape [2, E] — cat along dim=1
            cat_vals = torch.cat(values, dim=1)
        elif cat_dim == 0:
            cat_vals = torch.cat(values, dim=0)
        else:
            cat_vals = torch.cat(values, dim=cat_dim)

        # Build slices
        if key == "edge_index":
            sizes = torch.tensor([v.size(1) for v in values], dtype=torch.long)
        else:
            sizes = torch.tensor([v.size(0) for v in values], dtype=torch.long)

        slices = torch.zeros(len(values) + 1, dtype=torch.long)
        slices[1:] = torch.cumsum(sizes, dim=0)
        data_dict[key] = cat_vals
        slices_dict[key] = slices

    return data_dict, slices_dict


def _save(data_list, path):
    """Collate data_list and save to path."""
    data, slices = _collate(data_list)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save((data, slices), path)
    print(f"  Saved {len(data_list)} graphs → {path}")


# ---------------------------------------------------------------------------
# QM9
# ---------------------------------------------------------------------------

def create_qm9(root, n_molecules=100):
    """Create synthetic QM9 dataset.

    Each Data object has: x, z, pos, edge_index, smiles, edge_attr, y, name, idx
    """
    processed_path = os.path.join(root, "dataset/qm9/data/processed/data_v3.pt")

    atom_types = [1, 6, 7, 8, 9]  # H, C, N, O, F
    bond_types = 4  # single, double, triple, aromatic

    data_list = []
    for i in range(n_molecules):
        n_atoms = torch.randint(5, 20, (1,)).item()
        # One-hot atom type (5) + 6 features = 11-dim x
        type_idx = torch.randint(0, 5, (n_atoms,))
        one_hot = torch.zeros(n_atoms, 5)
        one_hot.scatter_(1, type_idx.unsqueeze(1), 1.0)

        other_feats = torch.randn(n_atoms, 6)  # atomic_number, aromatic, sp, sp2, sp3, num_hs
        x = torch.cat([one_hot, other_feats], dim=-1)

        z = torch.tensor([atom_types[t] for t in type_idx.tolist()], dtype=torch.long)
        pos = torch.randn(n_atoms, 3) * 2.0

        # Random sparse edges
        n_edges = max(n_atoms - 1, torch.randint(n_atoms - 1, 2 * n_atoms, (1,)).item())
        src = torch.randint(0, n_atoms, (n_edges,))
        dst = torch.randint(0, n_atoms, (n_edges,))
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

        # Edge attributes: one-hot bond type
        n_bonds = edge_index.size(1)
        et = torch.randint(0, bond_types, (n_bonds,))
        edge_attr = torch.zeros(n_bonds, bond_types)
        edge_attr.scatter_(1, et.unsqueeze(1), 1.0)

        y = torch.randn(1, 19)  # 19 regression targets

        data_list.append({
            "x": x,
            "z": z,
            "pos": pos,
            "edge_index": edge_index,
            "smiles": f"C{i}",  # placeholder
            "edge_attr": edge_attr,
            "y": y,
            "name": f"qm9_{i}",
            "idx": torch.tensor([i], dtype=torch.long),
        })

    # string fields can't be tensor-collated easily — store as list
    # PyG stores strings in __key__ or as a separate mechanism;
    # for simplicity, convert smiles/name to indices we can store
    for d in data_list:
        d.pop("smiles", None)
        d.pop("name", None)

    _save(data_list, processed_path)


# ---------------------------------------------------------------------------
# PI Repeat Unit
# ---------------------------------------------------------------------------

def create_pi_repeat_unit(root, n_units=20, embedding_dim=768):
    """Create synthetic PI repeat unit dataset.

    Each Data object has: x, z, edge_index, edge_type, embedding, name
    """
    processed_path = os.path.join(root, "dataset/pi/data/processed/data_repeat_unit.pt")

    atom_vocab = [0, 1, 6, 7, 8, 9, 16]  # includes dummy 0
    atom2id = {a: i for i, a in enumerate(atom_vocab)}

    data_list = []
    for i in range(n_units):
        n_atoms = torch.randint(15, 60, (1,)).item()

        # Random atoms from the PI element set (H, C, N, O, F, S)
        elements = [1, 6, 7, 8, 9, 16]
        atomic_numbers = [elements[torch.randint(0, len(elements), (1,)).item()] for _ in range(n_atoms)]

        z = torch.tensor(atomic_numbers, dtype=torch.long)
        x = torch.tensor([atom2id.get(a, 0) for a in atomic_numbers], dtype=torch.long)

        # Random edges (bonds)
        n_bonds = max(n_atoms - 1, torch.randint(n_atoms - 1, 2 * n_atoms, (1,)).item())
        src = torch.randint(0, n_atoms, (n_bonds,))
        dst = torch.randint(0, n_atoms, (n_bonds,))
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_type = torch.randint(0, 4, (edge_index.size(1),), dtype=torch.long)

        # polyBERT embedding
        embedding = torch.randn(1, embedding_dim)

        # name as tensor (store index, decode later if needed)
        name = torch.tensor([i], dtype=torch.long)

        data_list.append({
            "x": x,
            "z": z,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "embedding": embedding,
            "name": name,
        })

    _save(data_list, processed_path)

    # Return info needed for PI dataset
    unit_names = [f"diamine_{i//2}_dianhydride_{i%2}" for i in range(n_units)]
    unit_embeddings = [d["embedding"] for d in data_list]
    return unit_names, unit_embeddings


# ---------------------------------------------------------------------------
# PI (full)
# ---------------------------------------------------------------------------

def create_pi(root, unit_names, unit_embeddings, n_conformations_per_unit=3):
    """Create synthetic PI dataset.

    Each Data object has: x, z, edge_index, name, unit_embedding, degree, temperature, pos
    """
    processed_path = os.path.join(root, "dataset/pi/data/processed/data.pt")

    atom_vocab = [1, 6, 7, 8, 9, 16]
    atom2id = {a: i for i, a in enumerate(atom_vocab)}
    degrees = [16, 32, 64]
    temperatures = list(range(200, 650, 50))  # 9 values

    data_list = []
    for u_idx, (uname, uemb) in enumerate(zip(unit_names, unit_embeddings)):
        # Only use a subset for test data
        for degree in degrees[:2]:  # 16, 32
            for temp in temperatures[:3]:  # 200, 250, 300
                n_atoms = degree * torch.randint(20, 40, (1,)).item()
                n_atoms = min(n_atoms, 720)  # max_size from config

                elements = [atom_vocab[torch.randint(0, len(atom_vocab), (1,)).item()] for _ in range(n_atoms)]
                z = torch.tensor(elements, dtype=torch.long)
                x = torch.tensor([atom2id[a] for a in elements], dtype=torch.long)

                # Sparse edge_index for large graphs
                n_bonds = n_atoms * 2
                src = torch.randint(0, n_atoms, (n_bonds,))
                dst = torch.randint(0, n_atoms, (n_bonds,))
                mask = src != dst
                src, dst = src[mask], dst[mask]
                edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

                pos = torch.randn(n_atoms, 3) * 10.0

                data_list.append({
                    "x": x,
                    "z": z,
                    "edge_index": edge_index,
                    "name": torch.tensor([u_idx], dtype=torch.long),
                    "unit_embedding": uemb,
                    "degree": torch.tensor([degree], dtype=torch.int),
                    "temperature": torch.tensor([temp], dtype=torch.float),
                    "pos": pos,
                })

    _save(data_list, processed_path)
    return data_list


# ---------------------------------------------------------------------------
# PI Latent
# ---------------------------------------------------------------------------

def create_pi_latent(root, pi_data_list, code_dim=1024):
    """Create synthetic PI latent dataset.

    Adds a 'code' field to each PI Data object.
    Also creates code_stats.pt with mean and std.
    """
    processed_path = os.path.join(root, "dataset/pi/data/processed/data_with_latent.pt")
    stats_path = os.path.join(root, "dataset/pi/data/processed/code_stats.pt")

    latent_data_list = []
    all_codes = []
    for d in pi_data_list:
        code = torch.randn(1, code_dim)
        all_codes.append(code)
        new_d = dict(d)
        new_d["code"] = code
        latent_data_list.append(new_d)

    _save(latent_data_list, processed_path)

    # Stats
    codes_cat = torch.cat(all_codes, dim=0)  # [N, code_dim]
    stats = {
        "mean": codes_cat.mean(dim=0, keepdim=True),
        "std": codes_cat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6),
    }
    torch.save(stats, stats_path)
    print(f"  Saved code stats → {stats_path}")


# ---------------------------------------------------------------------------
# SDBS
# ---------------------------------------------------------------------------

def create_sdbs(root, n_molecules=80):
    """Create synthetic SDBS dataset.

    Each Data object has: x, z, edge_index, edge_attr, pos, smiles
    """
    processed_path = os.path.join(root, "dataset/sdbs/data/processed/data.pt")

    # 72 element types from the config
    atom_vocab = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40,
                  41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60,
                  62, 63, 64, 65, 66, 67, 68, 70, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83,
                  90, 92]
    atom2id = {a: i for i, a in enumerate(atom_vocab)}
    bond_types = 4

    data_list = []
    for i in range(n_molecules):
        n_atoms = torch.randint(5, 50, (1,)).item()

        # Most SDBS molecules are organic, so bias toward lighter elements
        common = [1, 6, 7, 8, 9, 16]
        atomic_numbers = [common[torch.randint(0, len(common), (1,)).item()] for _ in range(n_atoms)]

        z = torch.tensor(atomic_numbers, dtype=torch.long)
        x = torch.tensor([atom2id[a] for a in atomic_numbers], dtype=torch.long)

        # Random edges
        n_bonds = max(n_atoms - 1, torch.randint(n_atoms - 1, 2 * n_atoms, (1,)).item())
        src = torch.randint(0, n_atoms, (n_bonds,))
        dst = torch.randint(0, n_atoms, (n_bonds,))
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

        et = torch.randint(0, bond_types, (edge_index.size(1),))
        edge_attr = torch.zeros(edge_index.size(1), bond_types)
        edge_attr.scatter_(1, et.unsqueeze(1), 1.0)

        pos = torch.randn(n_atoms, 3) * 3.0

        data_list.append({
            "x": x,
            "z": z,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "pos": pos,
        })

    _save(data_list, processed_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test datasets for molgen")
    parser.add_argument("--root", default=".", help="Root directory (default: current dir)")
    parser.add_argument("--n-qm9", type=int, default=100, help="Number of QM9 molecules")
    parser.add_argument("--n-pi-units", type=int, default=20, help="Number of PI repeat units")
    parser.add_argument("--n-sdbs", type=int, default=80, help="Number of SDBS molecules")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print(f"Generating test datasets under: {root}")

    print("\n[1/5] QM9 dataset ...")
    create_qm9(root, n_molecules=args.n_qm9)

    print("\n[2/5] PI Repeat Unit dataset ...")
    unit_names, unit_embeddings = create_pi_repeat_unit(root, n_units=args.n_pi_units)

    print("\n[3/5] PI dataset ...")
    pi_data_list = create_pi(root, unit_names, unit_embeddings)

    print("\n[4/5] PI Latent dataset ...")
    create_pi_latent(root, pi_data_list)

    print("\n[5/5] SDBS dataset ...")
    create_sdbs(root, n_molecules=args.n_sdbs)

    print("\n✓ All test datasets generated.")
    print(f"  QM9:       {root}/dataset/qm9/data/processed/data_v3.pt")
    print(f"  PI unit:    {root}/dataset/pi/data/processed/data_repeat_unit.pt")
    print(f"  PI:         {root}/dataset/pi/data/processed/data.pt")
    print(f"  PI latent:  {root}/dataset/pi/data/processed/data_with_latent.pt")
    print(f"  PI stats:   {root}/dataset/pi/data/processed/code_stats.pt")
    print(f"  SDBS:       {root}/dataset/sdbs/data/processed/data.pt")


if __name__ == "__main__":
    main()
