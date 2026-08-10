# ============================================================
# EEI6373 - PERFORMANCE MODELLING MINI PROJECT
# THERMOPLASTIC MANUFACTURING SYSTEM
#
# Complete Performance Analysis + What-If Analysis
# + Discrete Event Simulation + Graph Generation
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

# Use non-interactive backend.
# This prevents plt.show() / Tkinter problems on Windows.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
import simpy


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Automatically locate the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Dataset should be in the same folder as this Python file
DATA_FILE = BASE_DIR / "Thermo_plastic_tc00234.csv"

# Folder where all analysis results will be stored
OUTPUT_DIR = BASE_DIR / "thermoplastic_performance_results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Folder for graphs
GRAPH_DIR = OUTPUT_DIR / "graphs"

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. START PROGRAM
# ============================================================

print("=" * 75)
print("EEI6373 - THERMOPLASTIC MANUFACTURING")
print("PERFORMANCE MODELING AND ANALYSIS")
print("=" * 75)

print("\nPython directory:")
print(BASE_DIR)

print("\nDataset:")
print(DATA_FILE)

print("\nDataset exists:", DATA_FILE.exists())


# ============================================================
# 4. CHECK DATASET
# ============================================================

if not DATA_FILE.exists():

    print("\nERROR:")
    print("The CSV file was not found.")

    print("\nExpected location:")
    print(DATA_FILE)

    print("\nMake sure this file exists:")
    print("Thermo_plastic_tc00234.csv")

    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")

print(
    "Number of production jobs:",
    len(df)
)

print(
    "Number of variables:",
    len(df.columns)
)


# ============================================================
# 6. REQUIRED COLUMNS
# ============================================================

required_columns = [

    "Arrival_Timestamp",

    "Batch_Quantity",

    "S1_Wait_Min",
    "S1_Service_Min",

    "S2_Wait_Min",
    "S2_Service_Min",

    "S3_Wait_Min",
    "S3_Service_Min",

    "S4_Wait_Min",
    "S4_Service_Min",

    "Total_Wait_Min",
    "Total_Service_Min",

    "Total_Flow_Min",

    "Scrap_Pct"
]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns
]


if missing_columns:

    print("\nERROR: Missing required columns:")

    for column in missing_columns:
        print(" -", column)

    raise ValueError(
        "Dataset does not contain all required columns."
    )


# ============================================================
# 7. DATA PREPARATION
# ============================================================

df["Arrival_Timestamp"] = pd.to_datetime(
    df["Arrival_Timestamp"],
    errors="coerce"
)


# Remove rows with invalid arrival timestamps

invalid_dates = (
    df["Arrival_Timestamp"].isna().sum()
)

if invalid_dates > 0:

    print(
        f"\nRemoving {invalid_dates} rows "
        "with invalid timestamps."
    )

    df = df.dropna(
        subset=["Arrival_Timestamp"]
    )


# Sort by arrival

df = df.sort_values(
    "Arrival_Timestamp"
).reset_index(drop=True)


# ============================================================
# 8. INTER-ARRIVAL TIME
# ============================================================

df["Inter_Arrival_Min"] = (

    df["Arrival_Timestamp"]
    .diff()
    .dt.total_seconds()
    / 60

)

# First job has no previous arrival

df["Inter_Arrival_Min"] = (
    df["Inter_Arrival_Min"]
    .fillna(0)
)


# ============================================================
# 9. BASIC DATA QUALITY ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("DATA QUALITY ANALYSIS")
print("=" * 75)


missing_summary = (

    df.isnull()
    .sum()
    .reset_index()

)

missing_summary.columns = [
    "Column",
    "Missing_Values"
]


print("\nMissing values:")

print(
    missing_summary.to_string(
        index=False
    )
)


# Save missing-value analysis

missing_summary.to_csv(

    OUTPUT_DIR
    / "01_missing_value_analysis.csv",

    index=False
)


# ============================================================
# 10. CHECK NEGATIVE VALUES
# ============================================================

time_columns = [

    "Inter_Arrival_Min",

    "S1_Wait_Min",
    "S1_Service_Min",

    "S2_Wait_Min",
    "S2_Service_Min",

    "S3_Wait_Min",
    "S3_Service_Min",

    "S4_Wait_Min",
    "S4_Service_Min",

    "Total_Wait_Min",
    "Total_Service_Min",
    "Total_Flow_Min"

]


negative_results = []


for column in time_columns:

    negative_count = (
        df[column] < 0
    ).sum()

    negative_results.append({

        "Column": column,

        "Negative_Count":
            negative_count

    })


negative_df = pd.DataFrame(
    negative_results
)


negative_df.to_csv(

    OUTPUT_DIR
    / "02_negative_value_analysis.csv",

    index=False
)


# ============================================================
# 11. VALIDATE TOTAL TIMES
# ============================================================

df["Calculated_Wait_Min"] = (

    df["S1_Wait_Min"]
    + df["S2_Wait_Min"]
    + df["S3_Wait_Min"]
    + df["S4_Wait_Min"]

)


df["Calculated_Service_Min"] = (

    df["S1_Service_Min"]
    + df["S2_Service_Min"]
    + df["S3_Service_Min"]
    + df["S4_Service_Min"]

)


df["Calculated_Flow_Min"] = (

    df["Calculated_Wait_Min"]
    + df["Calculated_Service_Min"]

)


df["Wait_Error"] = (

    df["Total_Wait_Min"]
    -
    df["Calculated_Wait_Min"]

)


df["Service_Error"] = (

    df["Total_Service_Min"]
    -
    df["Calculated_Service_Min"]

)


df["Flow_Error"] = (

    df["Total_Flow_Min"]
    -
    df["Calculated_Flow_Min"]

)


validation_results = pd.DataFrame({

    "Metric": [

        "Maximum Wait Error",

        "Maximum Service Error",

        "Maximum Flow Error"

    ],

    "Maximum_Absolute_Error": [

        df["Wait_Error"].abs().max(),

        df["Service_Error"].abs().max(),

        df["Flow_Error"].abs().max()

    ]

})


print("\nTime validation:")

print(
    validation_results.round(4)
    .to_string(index=False)
)


validation_results.to_csv(

    OUTPUT_DIR
    / "03_time_validation.csv",

    index=False
)


# ============================================================
# 12. OVERALL PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 75)
print("OVERALL PERFORMANCE METRICS")
print("=" * 75)


total_jobs = len(df)

mean_interarrival = (
    df["Inter_Arrival_Min"].iloc[1:].mean()
    if len(df) > 1
    else 0
)


mean_wait = (
    df["Total_Wait_Min"].mean()
)


mean_service = (
    df["Total_Service_Min"].mean()
)


mean_flow = (
    df["Total_Flow_Min"].mean()
)


median_flow = (
    df["Total_Flow_Min"].median()
)


p90_flow = (
    df["Total_Flow_Min"].quantile(0.90)
)


p95_flow = (
    df["Total_Flow_Min"].quantile(0.95)
)


mean_scrap = (
    df["Scrap_Pct"].mean()
)


# ============================================================
# 13. THROUGHPUT
# ============================================================

start_time = (
    df["Arrival_Timestamp"].min()
)

end_time = (
    df["Arrival_Timestamp"].max()
)


observation_hours = (

    (
        end_time
        -
        start_time
    ).total_seconds()
    / 3600

)


if observation_hours > 0:

    throughput_jobs_per_hour = (
        total_jobs
        /
        observation_hours
    )

else:

    throughput_jobs_per_hour = 0


throughput_jobs_per_day = (
    throughput_jobs_per_hour
    * 24
)


print(
    f"\nNumber of jobs: {total_jobs}"
)

print(
    f"Observation period: "
    f"{observation_hours:.2f} hours"
)

print(
    f"Throughput: "
    f"{throughput_jobs_per_hour:.2f} jobs/hour"
)

print(
    f"Average waiting time: "
    f"{mean_wait:.2f} minutes"
)

print(
    f"Average service time: "
    f"{mean_service:.2f} minutes"
)

print(
    f"Average flow time: "
    f"{mean_flow:.2f} minutes"
)

print(
    f"Average scrap: "
    f"{mean_scrap:.2f}%"
)


overall_results = pd.DataFrame({

    "Metric": [

        "Number of Jobs",

        "Observation Hours",

        "Throughput Jobs Per Hour",

        "Throughput Jobs Per Day",

        "Mean Inter Arrival Time",

        "Mean Waiting Time",

        "Mean Service Time",

        "Mean Flow Time",

        "Median Flow Time",

        "P90 Flow Time",

        "P95 Flow Time",

        "Mean Scrap Percentage"

    ],

    "Value": [

        total_jobs,

        observation_hours,

        throughput_jobs_per_hour,

        throughput_jobs_per_day,

        mean_interarrival,

        mean_wait,

        mean_service,

        mean_flow,

        median_flow,

        p90_flow,

        p95_flow,

        mean_scrap

    ]

})


overall_results.to_csv(

    OUTPUT_DIR
    / "04_overall_performance_results.csv",

    index=False
)


# ============================================================
# 14. STATION PERFORMANCE
# ============================================================

stations = [
    "S1",
    "S2",
    "S3",
    "S4"
]


station_results = []


for station in stations:

    wait_column = (
        f"{station}_Wait_Min"
    )

    service_column = (
        f"{station}_Service_Min"
    )


    total_wait_station = (
        df[wait_column].sum()
    )


    total_service_station = (
        df[service_column].sum()
    )


    station_results.append({

        "Station":
            station,

        "Mean_Wait_Min":
            df[wait_column].mean(),

        "Median_Wait_Min":
            df[wait_column].median(),

        "P90_Wait_Min":
            df[wait_column].quantile(0.90),

        "P95_Wait_Min":
            df[wait_column].quantile(0.95),

        "Mean_Service_Min":
            df[service_column].mean(),

        "Median_Service_Min":
            df[service_column].median(),

        "P90_Service_Min":
            df[service_column].quantile(0.90),

        "P95_Service_Min":
            df[service_column].quantile(0.95),

        "Total_Wait_Min":
            total_wait_station,

        "Total_Service_Min":
            total_service_station

    })


station_df = pd.DataFrame(
    station_results
)


# Percentage contribution

station_df["Wait_Share_Pct"] = (

    station_df["Total_Wait_Min"]
    /
    df["Total_Wait_Min"].sum()
    * 100

)


station_df["Service_Share_Pct"] = (

    station_df["Total_Service_Min"]
    /
    df["Total_Service_Min"].sum()
    * 100

)


station_df.to_csv(

    OUTPUT_DIR
    / "05_station_performance_results.csv",

    index=False
)


# ============================================================
# 15. BOTTLENECK RANKING
# ============================================================

station_df["Bottleneck_Score"] = (

    station_df["Mean_Wait_Min"]
    /
    station_df["Mean_Wait_Min"].max()
    * 100

)


bottleneck_df = (

    station_df[
        [
            "Station",
            "Mean_Wait_Min",
            "Mean_Service_Min",
            "Wait_Share_Pct",
            "Service_Share_Pct",
            "Bottleneck_Score"
        ]
    ]

    .sort_values(
        "Bottleneck_Score",
        ascending=False
    )

    .reset_index(drop=True)

)


bottleneck_df[
    "Bottleneck_Rank"
] = (

    bottleneck_df.index
    + 1

)


bottleneck_df.to_csv(

    OUTPUT_DIR
    / "06_bottleneck_ranking.csv",

    index=False
)


waiting_bottleneck = station_df.loc[

    station_df[
        "Mean_Wait_Min"
    ].idxmax()

]


service_bottleneck = station_df.loc[

    station_df[
        "Mean_Service_Min"
    ].idxmax()

]


print("\n" + "=" * 75)
print("BOTTLENECK ANALYSIS")
print("=" * 75)


print(
    f"\nWaiting-time bottleneck: "
    f"{waiting_bottleneck['Station']}"
)


print(
    f"Mean waiting time: "
    f"{waiting_bottleneck['Mean_Wait_Min']:.2f} minutes"
)


print(
    f"\nService-time bottleneck: "
    f"{service_bottleneck['Station']}"
)


print(
    f"Mean service time: "
    f"{service_bottleneck['Mean_Service_Min']:.2f} minutes"
)


# ============================================================
# 16. BATCH SIZE ANALYSIS
# ============================================================

df["Batch_Group"] = pd.qcut(

    df["Batch_Quantity"],

    q=4,

    duplicates="drop"

)


batch_analysis = (

    df.groupby(

        "Batch_Group",

        observed=True

    )

    .agg(

        Mean_Batch_Quantity=(
            "Batch_Quantity",
            "mean"
        ),

        Mean_Waiting_Time=(
            "Total_Wait_Min",
            "mean"
        ),

        Mean_Service_Time=(
            "Total_Service_Min",
            "mean"
        ),

        Mean_Flow_Time=(
            "Total_Flow_Min",
            "mean"
        ),

        Mean_Scrap_Pct=(
            "Scrap_Pct",
            "mean"
        ),

        Number_of_Jobs=(
            "Batch_Quantity",
            "count"
        )

    )

    .reset_index()

)


batch_analysis.to_csv(

    OUTPUT_DIR
    / "07_batch_size_analysis.csv",

    index=False
)


# ============================================================
# 17. CORRELATION ANALYSIS
# ============================================================

correlation_columns = [

    "Batch_Quantity",

    "Inter_Arrival_Min",

    "Total_Wait_Min",

    "Total_Service_Min",

    "Total_Flow_Min",

    "Scrap_Pct",

    "S1_Wait_Min",
    "S2_Wait_Min",
    "S3_Wait_Min",
    "S4_Wait_Min",

    "S1_Service_Min",
    "S2_Service_Min",
    "S3_Service_Min",
    "S4_Service_Min"

]


correlation_matrix = (

    df[
        correlation_columns
    ]

    .corr()

)


correlation_matrix.to_csv(

    OUTPUT_DIR
    / "08_correlation_matrix.csv"

)


# ============================================================
# 18. IMPORTANT PERFORMANCE RELATIONSHIPS
# ============================================================

batch_flow_corr = (

    df[
        "Batch_Quantity"
    ]

    .corr(

        df[
            "Total_Flow_Min"
        ]

    )

)


batch_wait_corr = (

    df[
        "Batch_Quantity"
    ]

    .corr(

        df[
            "Total_Wait_Min"
        ]

    )

)


batch_scrap_corr = (

    df[
        "Batch_Quantity"
    ]

    .corr(

        df[
            "Scrap_Pct"
        ]

    )

)


print("\n" + "=" * 75)
print("CORRELATION ANALYSIS")
print("=" * 75)


print(
    f"\nBatch quantity vs flow time: "
    f"{batch_flow_corr:.3f}"
)


print(
    f"Batch quantity vs waiting time: "
    f"{batch_wait_corr:.3f}"
)


print(
    f"Batch quantity vs scrap: "
    f"{batch_scrap_corr:.3f}"
)


# ============================================================
# 19. WHAT-IF ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("WHAT-IF PERFORMANCE IMPROVEMENT ANALYSIS")
print("=" * 75)


scenario_results = []


# ------------------------------------------------------------
# Scenario 1 - Baseline
# ------------------------------------------------------------

scenario_results.append({

    "Scenario":
        "Baseline",

    "Mean_Wait":
        df["Total_Wait_Min"].mean(),

    "Mean_Service":
        df["Total_Service_Min"].mean(),

    "Mean_Flow":
        df["Total_Flow_Min"].mean(),

    "Improvement_Pct":
        0

})


# ------------------------------------------------------------
# Scenario 2 - S2 waiting reduced by 25%
# ------------------------------------------------------------

s2_wait_improved = (

    df["Total_Wait_Min"]

    -

    df["S2_Wait_Min"]

    +

    df["S2_Wait_Min"] * 0.75

)


s2_flow_improved = (

    s2_wait_improved

    +

    df["Total_Service_Min"]

)


s2_flow_mean = (
    s2_flow_improved.mean()
)


s2_improvement = (

    (
        mean_flow
        -
        s2_flow_mean
    )
    /
    mean_flow
    * 100

)


scenario_results.append({

    "Scenario":
        "S2 Waiting -25%",

    "Mean_Wait":
        s2_wait_improved.mean(),

    "Mean_Service":
        mean_service,

    "Mean_Flow":
        s2_flow_mean,

    "Improvement_Pct":
        s2_improvement

})


# ------------------------------------------------------------
# Scenario 3 - S3 service reduced by 15%
# ------------------------------------------------------------

s3_service_improved = (

    df["Total_Service_Min"]

    -

    df["S3_Service_Min"]

    +

    df["S3_Service_Min"] * 0.85

)


s3_flow_improved = (

    df["Total_Wait_Min"]

    +

    s3_service_improved

)


s3_flow_mean = (
    s3_flow_improved.mean()
)


s3_improvement = (

    (
        mean_flow
        -
        s3_flow_mean
    )
    /
    mean_flow
    * 100

)


scenario_results.append({

    "Scenario":
        "S3 Service -15%",

    "Mean_Wait":
        mean_wait,

    "Mean_Service":
        s3_service_improved.mean(),

    "Mean_Flow":
        s3_flow_mean,

    "Improvement_Pct":
        s3_improvement

})


# ------------------------------------------------------------
# Scenario 4 - Combined
# ------------------------------------------------------------

combined_flow = (

    s2_wait_improved

    +

    s3_service_improved

)


combined_flow_mean = (
    combined_flow.mean()
)


combined_improvement = (

    (
        mean_flow
        -
        combined_flow_mean
    )
    /
    mean_flow
    * 100

)


scenario_results.append({

    "Scenario":
        "Combined S2 + S3",

    "Mean_Wait":
        s2_wait_improved.mean(),

    "Mean_Service":
        s3_service_improved.mean(),

    "Mean_Flow":
        combined_flow_mean,

    "Improvement_Pct":
        combined_improvement

})


scenario_df = pd.DataFrame(
    scenario_results
)


scenario_df.to_csv(

    OUTPUT_DIR
    / "09_what_if_scenarios.csv",

    index=False
)


print(
    scenario_df.round(2)
    .to_string(index=False)
)


# ============================================================
# 20. DISCRETE EVENT SIMULATION
# ============================================================

print("\n" + "=" * 75)
print("DISCRETE EVENT SIMULATION")
print("=" * 75)


# IMPORTANT:
#
# Actual machine/operator capacity is not present in the CSV.
#
# Therefore this initial model assumes:
#
# S1 = 1 resource
# S2 = 1 resource
# S3 = 1 resource
# S4 = 1 resource
#
# This is an explicit modeling assumption.
#
# Replace these capacities with actual factory information
# if available.


def run_simulation(

    data,

    s2_wait_factor=1.0,

    s3_service_factor=1.0

):

    env = simpy.Environment()


    resources = {

        "S1":
            simpy.Resource(
                env,
                capacity=1
            ),

        "S2":
            simpy.Resource(
                env,
                capacity=1
            ),

        "S3":
            simpy.Resource(
                env,
                capacity=1
            ),

        "S4":
            simpy.Resource(
                env,
                capacity=1
            )

    }


    results = []


    def process_job(
        job_id,
        row
    ):

        total_wait = 0

        total_service = 0


        for station in stations:

            resource = resources[
                station
            ]


            wait_time = float(

                row[
                    f"{station}_Wait_Min"
                ]

            )


            service_time = float(

                row[
                    f"{station}_Service_Min"
                ]

            )


            # Improvement applied to S2

            if station == "S2":

                wait_time *= (
                    s2_wait_factor
                )


            # Improvement applied to S3

            if station == "S3":

                service_time *= (
                    s3_service_factor
                )


            # ------------------------------------------------
            # Waiting component
            # ------------------------------------------------

            yield env.timeout(
                max(
                    wait_time,
                    0
                )
            )


            total_wait += max(
                wait_time,
                0
            )


            # ------------------------------------------------
            # Resource queue
            # ------------------------------------------------

            request_start = env.now


            with resource.request() as request:

                yield request


                resource_wait = (

                    env.now
                    -
                    request_start

                )


                total_wait += (
                    resource_wait
                )


                # ------------------------------------------------
                # Processing
                # ------------------------------------------------

                yield env.timeout(

                    max(
                        service_time,
                        0
                    )

                )


                total_service += max(
                    service_time,
                    0
                )


        total_flow = (

            total_wait
            +
            total_service

        )


        results.append({

            "Job":
                job_id,

            "Total_Wait":
                total_wait,

            "Total_Service":
                total_service,

            "Total_Flow":
                total_flow

        })


    def arrival_process():

        for index, row in data.iterrows():

            interarrival = float(

                row[
                    "Inter_Arrival_Min"
                ]

            )


            interarrival = max(
                interarrival,
                0
            )


            yield env.timeout(
                interarrival
            )


            env.process(

                process_job(
                    index,
                    row
                )

            )


    env.process(
        arrival_process()
    )


    env.run()


    return pd.DataFrame(
        results
    )


# ============================================================
# 21. RUN SIMULATION SCENARIOS
# ============================================================

simulation_scenarios = {

    "Baseline": {

        "s2_wait_factor": 1.00,

        "s3_service_factor": 1.00

    },

    "S2 Waiting -25%": {

        "s2_wait_factor": 0.75,

        "s3_service_factor": 1.00

    },

    "S3 Service -15%": {

        "s2_wait_factor": 1.00,

        "s3_service_factor": 0.85

    },

    "Combined S2 + S3": {

        "s2_wait_factor": 0.75,

        "s3_service_factor": 0.85

    }

}


simulation_results = []


for scenario, parameters in (

    simulation_scenarios.items()

):


    print(
        f"Running: {scenario}"
    )


    result = run_simulation(

        df,

        s2_wait_factor=
            parameters[
                "s2_wait_factor"
            ],

        s3_service_factor=
            parameters[
                "s3_service_factor"
            ]

    )


    simulation_results.append({

        "Scenario":
            scenario,

        "Mean_Wait":
            result[
                "Total_Wait"
            ].mean(),

        "Mean_Service":
            result[
                "Total_Service"
            ].mean(),

        "Mean_Flow":
            result[
                "Total_Flow"
            ].mean(),

        "P95_Flow":
            result[
                "Total_Flow"
            ].quantile(0.95)

    })


simulation_df = pd.DataFrame(
    simulation_results
)


simulation_baseline = (

    simulation_df.loc[

        simulation_df[
            "Scenario"
        ]
        ==
        "Baseline",

        "Mean_Flow"

    ].iloc[0]

)


simulation_df[
    "Flow_Improvement_Pct"
] = (

    (
        simulation_baseline
        -
        simulation_df[
            "Mean_Flow"
        ]
    )
    /
    simulation_baseline
    * 100

)


simulation_df.to_csv(

    OUTPUT_DIR
    / "10_simulation_results.csv",

    index=False
)


print("\nSimulation results:")

print(

    simulation_df.round(2)
    .to_string(index=False)

)


# ============================================================
# 22. THROUGHPUT BY TIME PERIOD
# ============================================================

# Divide the observation period into hourly groups

df["Arrival_Hour"] = (

    df[
        "Arrival_Timestamp"
    ]

    .dt.floor("h")

)


throughput_hourly = (

    df.groupby(
        "Arrival_Hour"
    )

    .size()

    .reset_index(
        name="Jobs_Completed"
    )

)


throughput_hourly.to_csv(

    OUTPUT_DIR
    / "11_hourly_throughput.csv",

    index=False
)


# ============================================================
# 23. GRAPH GENERATION
# ============================================================

print("\n" + "=" * 75)
print("GENERATING PERFORMANCE GRAPHS")
print("=" * 75)


# ------------------------------------------------------------
# GRAPH 01
# Station Waiting Time
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.bar(

    station_df[
        "Station"
    ],

    station_df[
        "Mean_Wait_Min"
    ]

)

plt.xlabel(
    "Production Station"
)

plt.ylabel(
    "Mean Waiting Time (minutes)"
)

plt.title(
    "Mean Waiting Time by Production Station"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "01_station_waiting_time_bar.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 02
# Station Service Time
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.bar(

    station_df[
        "Station"
    ],

    station_df[
        "Mean_Service_Min"
    ]

)

plt.xlabel(
    "Production Station"
)

plt.ylabel(
    "Mean Service Time (minutes)"
)

plt.title(
    "Mean Service Time by Production Station"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "02_station_service_time_bar.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 03
# Waiting vs Service
# ------------------------------------------------------------

x = np.arange(
    len(stations)
)

width = 0.35


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    x - width / 2,

    station_df[
        "Mean_Wait_Min"
    ],

    width,

    label="Waiting Time"

)


plt.bar(

    x + width / 2,

    station_df[
        "Mean_Service_Min"
    ],

    width,

    label="Service Time"

)


plt.xticks(
    x,
    stations
)

plt.xlabel(
    "Production Station"
)

plt.ylabel(
    "Time (minutes)"
)

plt.title(
    "Waiting Time vs Service Time"
)

plt.legend()

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "03_waiting_vs_service_bar.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 04
# Waiting Contribution Pie Chart
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 8)
)

plt.pie(

    station_df[
        "Total_Wait_Min"
    ],

    labels=station_df[
        "Station"
    ],

    autopct="%1.1f%%",

    startangle=90

)

plt.title(
    "Contribution of Each Station to Total Waiting Time"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "04_waiting_contribution_pie.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 05
# Service Contribution Pie Chart
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 8)
)

plt.pie(

    station_df[
        "Total_Service_Min"
    ],

    labels=station_df[
        "Station"
    ],

    autopct="%1.1f%%",

    startangle=90

)

plt.title(
    "Contribution of Each Station to Total Service Time"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "05_service_contribution_pie.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 06
# Flow Time Histogram
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.hist(

    df[
        "Total_Flow_Min"
    ],

    bins=20,

    edgecolor="black"

)

plt.axvline(

    mean_flow,

    linestyle="--",

    label="Mean Flow Time"

)

plt.axvline(

    p95_flow,

    linestyle=":",

    label="P95 Flow Time"

)

plt.xlabel(
    "Flow Time (minutes)"
)

plt.ylabel(
    "Number of Jobs"
)

plt.title(
    "Distribution of Total Flow Time"
)

plt.legend()

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "06_flow_time_histogram.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 07
# Waiting Time Histogram
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.hist(

    df[
        "Total_Wait_Min"
    ],

    bins=20,

    edgecolor="black"

)

plt.xlabel(
    "Total Waiting Time (minutes)"
)

plt.ylabel(
    "Number of Jobs"
)

plt.title(
    "Distribution of Total Waiting Time"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "07_waiting_time_histogram.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 08
# Service Time Histogram
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.hist(

    df[
        "Total_Service_Min"
    ],

    bins=20,

    edgecolor="black"

)

plt.xlabel(
    "Total Service Time (minutes)"
)

plt.ylabel(
    "Number of Jobs"
)

plt.title(
    "Distribution of Total Service Time"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "08_service_time_histogram.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 09
# Batch Quantity vs Flow Time
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.scatter(

    df[
        "Batch_Quantity"
    ],

    df[
        "Total_Flow_Min"
    ],

    alpha=0.65

)


z = np.polyfit(

    df[
        "Batch_Quantity"
    ],

    df[
        "Total_Flow_Min"
    ],

    1

)


x_line = np.linspace(

    df[
        "Batch_Quantity"
    ].min(),

    df[
        "Batch_Quantity"
    ].max(),

    100

)


plt.plot(

    x_line,

    z[0] * x_line + z[1],

    linestyle="--",

    label=
        f"Correlation = {batch_flow_corr:.3f}"

)


plt.xlabel(
    "Batch Quantity"
)

plt.ylabel(
    "Total Flow Time (minutes)"
)

plt.title(
    "Batch Quantity vs Total Flow Time"
)

plt.legend()

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "09_batch_vs_flow_scatter.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 10
# Batch Quantity vs Waiting
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.scatter(

    df[
        "Batch_Quantity"
    ],

    df[
        "Total_Wait_Min"
    ],

    alpha=0.65

)

plt.xlabel(
    "Batch Quantity"
)

plt.ylabel(
    "Total Waiting Time (minutes)"
)

plt.title(
    "Batch Quantity vs Waiting Time"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "10_batch_vs_waiting_scatter.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 11
# Batch Quantity vs Scrap
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.scatter(

    df[
        "Batch_Quantity"
    ],

    df[
        "Scrap_Pct"
    ],

    alpha=0.65

)

plt.xlabel(
    "Batch Quantity"
)

plt.ylabel(
    "Scrap Percentage (%)"
)

plt.title(
    "Batch Quantity vs Scrap Percentage"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "11_batch_vs_scrap_scatter.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 12
# Inter-arrival Distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.hist(

    df[
        "Inter_Arrival_Min"
    ].iloc[1:],

    bins=20,

    edgecolor="black"

)

plt.xlabel(
    "Inter-arrival Time (minutes)"
)

plt.ylabel(
    "Number of Jobs"
)

plt.title(
    "Distribution of Job Inter-arrival Times"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "12_interarrival_distribution.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 13
# Throughput Over Time
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(

    throughput_hourly[
        "Arrival_Hour"
    ],

    throughput_hourly[
        "Jobs_Completed"
    ],

    marker="o"

)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Number of Jobs"
)

plt.title(
    "Production Throughput Over Time"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "13_throughput_over_time.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 14
# Flow Time by Batch Group
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.bar(

    batch_analysis[
        "Batch_Group"
    ].astype(str),

    batch_analysis[
        "Mean_Flow_Time"
    ]

)

plt.xlabel(
    "Batch Size Group"
)

plt.ylabel(
    "Mean Flow Time (minutes)"
)

plt.title(
    "Flow Time by Batch Size Group"
)

plt.xticks(
    rotation=25
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "14_flow_by_batch_group.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 15
# Scrap by Batch Group
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.bar(

    batch_analysis[
        "Batch_Group"
    ].astype(str),

    batch_analysis[
        "Mean_Scrap_Pct"
    ]

)

plt.xlabel(
    "Batch Size Group"
)

plt.ylabel(
    "Mean Scrap (%)"
)

plt.title(
    "Scrap Percentage by Batch Size Group"
)

plt.xticks(
    rotation=25
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "15_scrap_by_batch_group.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 16
# Bottleneck Comparison
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.bar(

    bottleneck_df[
        "Station"
    ],

    bottleneck_df[
        "Bottleneck_Score"
    ]

)

plt.xlabel(
    "Station"
)

plt.ylabel(
    "Relative Bottleneck Score"
)

plt.title(
    "Station Bottleneck Ranking"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "16_bottleneck_ranking.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 17
# What-if Scenario Comparison
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.bar(

    scenario_df[
        "Scenario"
    ],

    scenario_df[
        "Mean_Flow"
    ]

)

plt.xlabel(
    "Scenario"
)

plt.ylabel(
    "Mean Flow Time (minutes)"
)

plt.title(
    "Baseline vs Proposed Performance Improvements"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "17_what_if_flow_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 18
# Simulation Comparison
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.bar(

    simulation_df[
        "Scenario"
    ],

    simulation_df[
        "Mean_Flow"
    ]

)

plt.xlabel(
    "Simulation Scenario"
)

plt.ylabel(
    "Mean Simulated Flow Time (minutes)"
)

plt.title(
    "Discrete-Event Simulation Comparison"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "18_simulation_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 19
# P95 Comparison
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.bar(

    simulation_df[
        "Scenario"
    ],

    simulation_df[
        "P95_Flow"
    ]

)

plt.xlabel(
    "Simulation Scenario"
)

plt.ylabel(
    "P95 Flow Time (minutes)"
)

plt.title(
    "95th Percentile Flow-Time Comparison"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "19_p95_flow_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 20
# Improvement Percentage
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.bar(

    simulation_df[
        "Scenario"
    ],

    simulation_df[
        "Flow_Improvement_Pct"
    ]

)

plt.xlabel(
    "Scenario"
)

plt.ylabel(
    "Flow Time Improvement (%)"
)

plt.title(
    "Performance Improvement Relative to Baseline"
)

plt.xticks(
    rotation=20
)

plt.axhline(
    0,
    linestyle="-"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "20_performance_improvement.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ------------------------------------------------------------
# GRAPH 21
# Correlation Heatmap
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 10)
)

plt.imshow(

    correlation_matrix,

    aspect="auto"

)

plt.colorbar(
    label="Correlation"
)


plt.xticks(

    range(
        len(correlation_matrix.columns)
    ),

    correlation_matrix.columns,

    rotation=90

)


plt.yticks(

    range(
        len(correlation_matrix.columns)
    ),

    correlation_matrix.columns

)


plt.title(
    "Performance Variable Correlation Matrix"
)

plt.tight_layout()

plt.savefig(

    GRAPH_DIR
    / "21_correlation_heatmap.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()


# ============================================================
# 24. SAVE CLEANED DATA
# ============================================================

df.to_csv(

    OUTPUT_DIR
    / "12_cleaned_analysis_dataset.csv",

    index=False

)


# ============================================================
# 25. CREATE FINAL PERFORMANCE SUMMARY
# ============================================================

final_summary = pd.DataFrame({

    "Performance_Metric": [

        "Total Production Jobs",

        "Throughput (jobs/hour)",

        "Average Inter-arrival Time (min)",

        "Average Waiting Time (min)",

        "Average Service Time (min)",

        "Average Flow Time (min)",

        "Median Flow Time (min)",

        "P90 Flow Time (min)",

        "P95 Flow Time (min)",

        "Average Scrap (%)",

        "Waiting Bottleneck",

        "Service Bottleneck",

        "Batch vs Flow Correlation"

    ],

    "Result": [

        total_jobs,

        throughput_jobs_per_hour,

        mean_interarrival,

        mean_wait,

        mean_service,

        mean_flow,

        median_flow,

        p90_flow,

        p95_flow,

        mean_scrap,

        waiting_bottleneck[
            "Station"
        ],

        service_bottleneck[
            "Station"
        ],

        batch_flow_corr

    ]

})


final_summary.to_csv(

    OUTPUT_DIR
    / "13_FINAL_PERFORMANCE_SUMMARY.csv",

    index=False

)


# ============================================================
# 26. CREATE README FOR RESULTS
# ============================================================

readme_text = f"""
EEI6373 PERFORMANCE MODELLING
THERMOPLASTIC MANUFACTURING SYSTEM

DATASET:
{DATA_FILE}

NUMBER OF JOBS:
{total_jobs}

AVERAGE WAITING TIME:
{mean_wait:.2f} minutes

AVERAGE SERVICE TIME:
{mean_service:.2f} minutes

AVERAGE FLOW TIME:
{mean_flow:.2f} minutes

P95 FLOW TIME:
{p95_flow:.2f} minutes

THROUGHPUT:
{throughput_jobs_per_hour:.2f} jobs/hour

AVERAGE SCRAP:
{mean_scrap:.2f} %

WAITING-TIME BOTTLENECK:
{waiting_bottleneck['Station']}

SERVICE-TIME BOTTLENECK:
{service_bottleneck['Station']}

BATCH VS FLOW CORRELATION:
{batch_flow_corr:.3f}

IMPORTANT SIMULATION ASSUMPTION:
One resource/server is assumed at each station S1-S4
because actual machine/operator capacity was not available
in the supplied dataset.

GENERATED FILES:

01_missing_value_analysis.csv
02_negative_value_analysis.csv
03_time_validation.csv
04_overall_performance_results.csv
05_station_performance_results.csv
06_bottleneck_ranking.csv
07_batch_size_analysis.csv
08_correlation_matrix.csv
09_what_if_scenarios.csv
10_simulation_results.csv
11_hourly_throughput.csv
12_cleaned_analysis_dataset.csv
13_FINAL_PERFORMANCE_SUMMARY.csv

GRAPHS:

01_station_waiting_time_bar.png
02_station_service_time_bar.png
03_waiting_vs_service_bar.png
04_waiting_contribution_pie.png
05_service_contribution_pie.png
06_flow_time_histogram.png
07_waiting_time_histogram.png
08_service_time_histogram.png
09_batch_vs_flow_scatter.png
10_batch_vs_waiting_scatter.png
11_batch_vs_scrap_scatter.png
12_interarrival_distribution.png
13_throughput_over_time.png
14_flow_by_batch_group.png
15_scrap_by_batch_group.png
16_bottleneck_ranking.png
17_what_if_flow_comparison.png
18_simulation_comparison.png
19_p95_flow_comparison.png
20_performance_improvement.png
21_correlation_heatmap.png
"""


with open(

    OUTPUT_DIR
    / "README_RESULTS.txt",

    "w",

    encoding="utf-8"

) as file:

    file.write(
        readme_text
    )


# ============================================================
# 27. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("PERFORMANCE MODELING ANALYSIS COMPLETED")
print("=" * 75)

print(
    "\nResults folder:"
)

print(
    OUTPUT_DIR
)

print(
    "\nGraphs folder:"
)

print(
    GRAPH_DIR
)

print("\nGenerated 21 performance graphs.")

print(
    "\nMain performance results:"
)

print(
    f"Throughput = "
    f"{throughput_jobs_per_hour:.2f} jobs/hour"
)

print(
    f"Mean Waiting Time = "
    f"{mean_wait:.2f} minutes"
)

print(
    f"Mean Service Time = "
    f"{mean_service:.2f} minutes"
)

print(
    f"Mean Flow Time = "
    f"{mean_flow:.2f} minutes"
)

print(
    f"P95 Flow Time = "
    f"{p95_flow:.2f} minutes"
)

print(
    f"Waiting Bottleneck = "
    f"{waiting_bottleneck['Station']}"
)

print(
    f"Service Bottleneck = "
    f"{service_bottleneck['Station']}"
)

print(
    f"Batch vs Flow Correlation = "
    f"{batch_flow_corr:.3f}"
)

print("\nAnalysis finished successfully.")