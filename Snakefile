rule all:
    input:
        "results/csvs/MiniRun6.5_tracks.csv",
        "results/csvs/MiniRun6.5_segments.csv",
        "results/plots/MiniRun6.5_purity.png",
        "results/plots/MiniRun6.5_lifetime.png"
rule selection:
    input:
        "data/MiniRun6.5_1E19_RHC.flow.0000433.FLOW.proto_nd_flow.hdf5"
    output:
        "results/csvs/MiniRun6.5_tracks.csv",
        "results/csvs/MiniRun6.5_segments.csv",
        "results/plots/MiniRun6.5_purity.png"
    shell:
        "python3 src/MuonSelection.py {input} {output}"

rule lifetime:
    input:
        "results/csvs/MiniRun6.5_segments.csv"
    output:
        "results/plots/MiniRun6.5_lifetime.png"
    shell:
        "python3 src/Lifetime.py {input} {output}"