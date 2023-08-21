# TO DO
# - Partial antibiotic cohort query (vitals, sepsis indicator, labs, all cause mortality)
# - dataframe to pytorch dataset
# - (optional) dataframe to csv files / loader function

import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas.io import gbq
import argparse
import sparse
import os

# query MIMIC-IV via BigQuery
def get_cohort(project, cohort):
    if cohort=='test':
        antibiotic_ids = [225798,225840,225842,225845,225847,225850,225859,225860,225865,225866,225875,225879,225881,225883,225886,225888,225889,225898,225902,227691,229587]

        query = """
        WITH pop AS (
          SELECT DISTINCT stay_id AS ID
          FROM `physionet-data.mimiciv_icu.icustays`
          ORDER BY stay_id
          LIMIT 1000
        ), data AS (
          SELECT
            c.stay_id AS ID,
            c.charttime AS t,
            d.label AS variable_name,
            c.valuenum AS variable_value
          FROM `physionet-data.mimiciv_icu.chartevents` AS c
          JOIN `physionet-data.mimiciv_icu.d_items` AS d ON c.itemid = d.itemid
          UNION ALL
          SELECT s.stay_id AS ID,
            s.suspected_infection_time AS t,
            CASE WHEN s.sepsis3=TRUE THEN "sepsis" ELSE "sepsis"
              END AS variable_name,
            CASE WHEN s.sepsis3=TRUE THEN 1 ELSE 0
              END AS variable_value
          FROM `physionet-data.mimiciv_derived.sepsis3` AS s
          UNION ALL
          SELECT
            i.stay_id AS ID,
            i.starttime AS t,
            CASE WHEN i.itemid IN ({}) THEN "antibiotic" ELSE "null"
              END AS variable_name,
            i.amount AS variable_value
          FROM `physionet-data.mimiciv_icu.inputevents` AS i
          WHERE i.itemid IN ({})
        )
        SELECT pop.ID, data.t, data.variable_name, data.variable_value
        FROM pop
        LEFT JOIN data ON pop.ID = data.ID
        """.format(','.join(map(str, antibiotic_ids)), ','.join(map(str, antibiotic_ids)))

        return gbq.read_gbq(query, project_id=project)

    # returns df with columns [ID, t, variable_name, variable_value]
    if cohort=='sepsis':
        antibiotic_ids = [225798,225840,225842,225845,225847,225850,225859,225860,225865,225866,225875,225879,225881,225883,225886,225888,225889,225898,225902,227691,229587]
        
        query = """
        SELECT
          c.stay_id AS ID,
          c.charttime AS t,
          d.label AS variable_name,
          c.valuenum AS variable_value
        FROM `physionet-data.mimiciv_icu.chartevents` AS c
        JOIN `physionet-data.mimiciv_icu.d_items` AS d ON c.itemid = d.itemid
        UNION ALL
        SELECT s.stay_id AS ID,
          s.suspected_infection_time AS t,
          CASE WHEN s.sepsis3=TRUE THEN "sepsis" ELSE "sepsis"
            END AS variable_name,
          CASE WHEN s.sepsis3=TRUE THEN 1 ELSE 0
            END AS variable_value
        FROM `physionet-data.mimiciv_derived.sepsis3` AS s
        UNION ALL
        SELECT
          i.stay_id AS ID,
          i.starttime AS t,
          CASE WHEN i.itemid IN ({}) THEN "antibiotic" ELSE "null"
            END AS variable_name,
          i.amount AS variable_value
        FROM `physionet-data.mimiciv_icu.inputevents` AS i
        WHERE i.itemid IN ({})
        """.format(','.join(map(str, antibiotic_ids)), ','.join(map(str, antibiotic_ids)))

        return gbq.read_gbq(query, project_id=project)
    
    if cohort=='nutrition':
        query = """
        WITH vent AS (
          SELECT stay_id,
            starttime AS vent_starttime,
            endtime AS vent_endtime,
            CASE WHEN 
              ventilation_status IN ('InvasiveVent') THEN 1 ELSE 0
            END AS vented
          FROM `physionet-data.mimiciv_derived.ventilation`
        ), vaso AS (
          SELECT v.stay_id,
            v.starttime AS vaso_starttime,
            v.endtime AS vaso_endtime,
            v.phenylephrine, v.epinephrine, v.vasopressin, v.norepinephrine, v.dobutamine, v.dopamine,
            COALESCE(v.phenylephrine/10, NULL) AS ned_phenylephrine,
            COALESCE(v.epinephrine, NULL) AS ned_epinephrine,
            COALESCE(v.vasopressin*2.5, NULL) AS ned_vasopressin,
            COALESCE(v.dopamine/100, NULL) AS ned_dopamine,
            COALESCE(v.norepinephrine, NULL) AS ned_norepinephrine,
            CASE WHEN
              COALESCE(v.phenylephrine/10, NULL) >= 0.5 OR
              COALESCE(v.epinephrine, NULL) >= 0.5 OR
              COALESCE(v.vasopressin*2.5, NULL) >= 0.5 OR
              COALESCE(v.dopamine/100, NULL) >= 0.5 OR
              COALESCE(v.norepinephrine, NULL) >= 0.5
            THEN 1 ELSE 0
            END AS vaso_high
          FROM `physionet-data.mimiciv_derived.vasoactive_agent` v
        ), ntrn AS (
          SELECT i.stay_id, i.rate AS en_rate,
            i.starttime AS en_starttime,
            i.endtime AS en_endtime,
            CASE WHEN
              i.rate >= 25 THEN 1 ELSE 0
            END AS en_high,
            CASE WHEN i.itemid = 221036 THEN 1 ELSE 0 END AS NutrenRenal_Full,
            CASE WHEN i.itemid = 221207 THEN 1 ELSE 0 END AS Impact_Full,
            CASE WHEN i.itemid = 225928 THEN 1 ELSE 0 END AS ImpactwithFiber_Full,
            CASE WHEN i.itemid = 225929 THEN 1 ELSE 0 END AS ProBalance_Full,
            CASE WHEN i.itemid = 225930 THEN 1 ELSE 0 END AS Peptamen1_5_Full,
            CASE WHEN i.itemid = 225931 THEN 1 ELSE 0 END AS Nutren2_0_Full,
            CASE WHEN i.itemid = 225934 THEN 1 ELSE 0 END AS Vivonex_Full,
            CASE WHEN i.itemid = 225935 THEN 1 ELSE 0 END AS Replete_Full,
            CASE WHEN i.itemid = 225936 THEN 1 ELSE 0 END AS RepletewithFiber_Full,
            CASE WHEN i.itemid = 225937 THEN 1 ELSE 0 END AS Ensure_Full,
            CASE WHEN i.itemid = 225970 THEN 1 ELSE 0 END AS Beneprotein,
            CASE WHEN i.itemid = 226016 THEN 1 ELSE 0 END AS Nutren2_0_1_4,
            CASE WHEN i.itemid = 226017 THEN 1 ELSE 0 END AS Nutren2_0_2_3,
            CASE WHEN i.itemid = 226019 THEN 1 ELSE 0 END AS Nutren2_0_1_2,
            CASE WHEN i.itemid = 226020 THEN 1 ELSE 0 END AS Impact_1_4,
            CASE WHEN i.itemid = 226022 THEN 1 ELSE 0 END AS Impact_3_4,
            CASE WHEN i.itemid = 226023 THEN 1 ELSE 0 END AS Impact_1_2,
            CASE WHEN i.itemid = 226024 THEN 1 ELSE 0 END AS ImpactwithFiber_1_4,
            CASE WHEN i.itemid = 226026 THEN 1 ELSE 0 END AS ImpactwithFiber_3_4,
            CASE WHEN i.itemid = 226027 THEN 1 ELSE 0 END AS ImpactwithFiber_1_2,
            CASE WHEN i.itemid = 226028 THEN 1 ELSE 0 END AS NutrenRenal_1_4,
            CASE WHEN i.itemid = 226030 THEN 1 ELSE 0 END AS NutrenRenal_3_4,
            CASE WHEN i.itemid = 226031 THEN 1 ELSE 0 END AS NutrenRenal_1_2,
            CASE WHEN i.itemid = 226036 THEN 1 ELSE 0 END AS Peptamen1_5_1_4,
            CASE WHEN i.itemid = 226038 THEN 1 ELSE 0 END AS Peptamen1_5_3_4,
            CASE WHEN i.itemid = 226039 THEN 1 ELSE 0 END AS Peptamen1_5_1_2,
            CASE WHEN i.itemid = 226044 THEN 1 ELSE 0 END AS Replete_1_4,
            CASE WHEN i.itemid = 226045 THEN 1 ELSE 0 END AS Replete_2_3,
            CASE WHEN i.itemid = 226046 THEN 1 ELSE 0 END AS Replete_3_4,
            CASE WHEN i.itemid = 226047 THEN 1 ELSE 0 END AS Replete_1_2,
            CASE WHEN i.itemid = 226048 THEN 1 ELSE 0 END AS RepletewithFiber_1_4,
            CASE WHEN i.itemid = 226049 THEN 1 ELSE 0 END AS RepletewithFiber_2_3,
            CASE WHEN i.itemid = 226050 THEN 1 ELSE 0 END AS RepletewithFiber_3_4,
            CASE WHEN i.itemid = 226051 THEN 1 ELSE 0 END AS RepletewithFiber_1_2,
            CASE WHEN i.itemid = 226058 THEN 1 ELSE 0 END AS Vivonex_3_4,
            CASE WHEN i.itemid = 226059 THEN 1 ELSE 0 END AS Vivonex_1_2,
            CASE WHEN i.itemid = 226875 THEN 1 ELSE 0 END AS Ensure_3_4,
            CASE WHEN i.itemid = 226877 THEN 1 ELSE 0 END AS EnsurePlus_Full,
            CASE WHEN i.itemid = 226880 THEN 1 ELSE 0 END AS NutrenPulmonary_Full,
            CASE WHEN i.itemid = 226881 THEN 1 ELSE 0 END AS NutrenPulmonary_3_4,
            CASE WHEN i.itemid = 226882 THEN 1 ELSE 0 END AS NutrenPulmonary_1_2,
            CASE WHEN i.itemid = 227518 THEN 1 ELSE 0 END AS Nutren2_0_3_4,
            CASE WHEN i.itemid = 227695 THEN 1 ELSE 0 END AS FibersourceHN_Full,
            CASE WHEN i.itemid = 227696 THEN 1 ELSE 0 END AS FibersourceHN_3_4,
            CASE WHEN i.itemid = 227698 THEN 1 ELSE 0 END AS FibersourceHN_1_2,
            CASE WHEN i.itemid = 227699 THEN 1 ELSE 0 END AS FibersourceHN_1_4,
            CASE WHEN i.itemid = 227973 THEN 1 ELSE 0 END AS NovaSourceRenal_1_2,
            CASE WHEN i.itemid = 227974 THEN 1 ELSE 0 END AS NovaSourceRenal_3_4,
            CASE WHEN i.itemid = 227975 THEN 1 ELSE 0 END AS NovaSourceRenal_Full,
            CASE WHEN i.itemid = 227976 THEN 1 ELSE 0 END AS BoostGlucoseControl_1_4,
            CASE WHEN i.itemid = 227977 THEN 1 ELSE 0 END AS BoostGlucoseControl_1_2,
            CASE WHEN i.itemid = 227978 THEN 1 ELSE 0 END AS BoostGlucoseControl_3_4,
            CASE WHEN i.itemid = 227979 THEN 1 ELSE 0 END AS BoostGlucoseControl_Full,
            CASE WHEN i.itemid = 228131 THEN 1 ELSE 0 END AS Isosource1_5_1_2,
            CASE WHEN i.itemid = 228132 THEN 1 ELSE 0 END AS Isosource1_5_1_4,
            CASE WHEN i.itemid = 228133 THEN 1 ELSE 0 END AS Isosource1_5_2_3,
            CASE WHEN i.itemid = 228134 THEN 1 ELSE 0 END AS Isosource1_5_3_4,
            CASE WHEN i.itemid = 228135 THEN 1 ELSE 0 END AS Isosource1_5_Full,
            CASE WHEN i.itemid = 228348 THEN 1 ELSE 0 END AS Nepro_1_2,
            CASE WHEN i.itemid = 228351 THEN 1 ELSE 0 END AS Nepro_Full,
            CASE WHEN i.itemid = 228355 THEN 1 ELSE 0 END AS Enlive_Full,
            CASE WHEN i.itemid = 228356 THEN 1 ELSE 0 END AS Glucerna_1_2,
            CASE WHEN i.itemid = 228359 THEN 1 ELSE 0 END AS Glucerna_Full,
            CASE WHEN i.itemid = 228360 THEN 1 ELSE 0 END AS Pulmocare_1_2,
            CASE WHEN i.itemid = 228361 THEN 1 ELSE 0 END AS Pulmocare_1_4,
            CASE WHEN i.itemid = 228363 THEN 1 ELSE 0 END AS Pulmocare_Full,
            CASE WHEN i.itemid = 228364 THEN 1 ELSE 0 END AS TwoCalHN_1_2,
            CASE WHEN i.itemid = 228367 THEN 1 ELSE 0 END AS TwoCalHN_Full,
            CASE WHEN i.itemid = 228383 THEN 1 ELSE 0 END AS PeptamenBariatric_Full,
            CASE WHEN i.itemid = 229009 THEN 1 ELSE 0 END AS Promote_Full,
            CASE WHEN i.itemid = 229010 THEN 1 ELSE 0 END AS Jevity1_2_Full,
            CASE WHEN i.itemid = 229011 THEN 1 ELSE 0 END AS Jevity1_5_Full,
            CASE WHEN i.itemid = 229012 THEN 1 ELSE 0 END AS Vital1_5_Full,
            CASE WHEN i.itemid = 229013 THEN 1 ELSE 0 END AS Glucerna1_2_Full,
            CASE WHEN i.itemid = 229014 THEN 1 ELSE 0 END AS PromotewithFiber_Full,
            CASE WHEN i.itemid = 229295 THEN 1 ELSE 0 END AS Glucerna1_5_Full,
            CASE WHEN i.itemid = 229296 THEN 1 ELSE 0 END AS VitalHighProtein_Full,
            CASE WHEN i.itemid = 229297 THEN 1 ELSE 0 END AS Osmolite1_5_Full
          FROM `physionet-data.mimiciv_icu.inputevents` AS i
          JOIN `physionet-data.mimiciv_icu.d_items` AS d ON d.itemid = i.itemid
          WHERE
            i.ordercategoryname LIKE'%13-Enteral Nutrition%'
            OR i.secondaryordercategoryname LIKE'%Additives (EN)'
        )
        SELECT icu.stay_id,
          icu.admission_age,
          icu_intime, icu_outtime,
          vent.*,
          vaso.*,
          ntrn.*
        FROM `physionet-data.mimiciv_derived.icustay_detail` AS icu
        INNER JOIN vent ON vent.stay_id = icu.stay_id
        INNER JOIN vaso ON vaso.stay_id = icu.stay_id
        JOIN ntrn ON ntrn.stay_id = icu.stay_id
        WHERE
          vented = 1
          AND admission_age >= 18
          AND icu_intime <= vaso_starttime
        ORDER BY icu.stay_id
        """
        return gbq.read_gbq(query, project_id=project)
    
def preprocess_mimic(project, cohort):
    # Input:
    #   - dataframe of cohort with columns [ID, t, variable name, variable value]
    #
    # Output: The generated features and associated metadata are located in {data_path}/
    #   -s.npz: a sparse array of shape (N, d)
    #   -X.npz: a sparse tensor of shape (N, L, D)
    #   -s.feature_names.json: names of d time-invariant features
    #   -X.feature_names.json: names of D time-series features
    #   -x.feature_aliases.json: aliases of duplicated time-invariant features
    #   -X.feature_aliases.json: aliases of duplicated time-series features

    print('Querying MIMIC-IV ...')
    df = get_cohort(project, cohort)
    test_pop = gbq.read_gbq("""
        SELECT distinct stay_id AS ID
        FROM `physionet-data.mimiciv_icu.icustays`
        ORDER BY stay_id
        LIMIT 1000
        """, project_id=project)
    
    df.to_csv('small_test/input/input_data.csv')
    test_pop.to_csv('small_test/input/pop.csv')
    print('Done!')

    

# Create PyTorch dataset from queried data
class MIMICDataset(Dataset):
    def __init__(self, project, cohort):
        self.project = project
        self.cohort = cohort

    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def make_dataset(self, project, cohort):
        # query data
        df = get_cohort(project=project, cohort=cohort)

        # separate into dataframes per individual
        for i, id in enumerate(df['stay_id'].unique()):
            sub_df = df[df['stay_id']==id]


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # user arguments
    parser.add_argument('--project', type=str, default='mimic-380019', help='Project ID for Google Cloud project hosting MIMIC-IV data')
    parser.add_argument('--cohort', type=str, default='sepsis', help='Cohort (sepsis or nutrition) from experiments to select')
    parser.add_argument('--data_path', type=str, default="None", help='path to download data to')
    parser.add_argument('--seed', type=int, default=387923)

    args = parser.parse_args()

    # run functionality
    torch.manual_seed(args.seed)

    # query and format data for FIDDLE input
    preprocess_mimic(project=args.project, cohort=args.cohort)