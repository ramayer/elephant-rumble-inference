import csv
import duckdb
import glob
import os
import re
import torch
from dataclasses import dataclass
from typing import Optional
# shorter names for important columns from the raven file
@dataclass
class RavenLabel:
    bt: float
    et: float
    lf: float
    hf: float
    duration: float
    audio_file: str
    t1: Optional[str]
    t2: Optional[str]
    t3: Optional[str]
    notes: Optional[str]
    score: Optional[str]
    ravenfile: Optional[str]

class RavenFileHelper:
    def __init__(self,root_path=None):
        self.ddb = duckdb.connect()
        if root_path:
            self.root_path = root_path
            self.raven_files = self.find_candidate_raven_files(root_path)
            self.all_raven_data = self.all_raven_files_as_one_table(self.raven_files)

    def find_continuous_segments(self, boolean_tensor):
        sign_changes = torch.cat(
            [
                torch.tensor([True]),
                boolean_tensor[1:] != boolean_tensor[:-1],
                torch.tensor([True]),
            ]
        )
        change_indices = torch.where(sign_changes)[0]
        segments = []
        for start, end in zip(change_indices[:-1], change_indices[1:]):
            if boolean_tensor[start]:
                segments.append((start.item(), end.item() - 1))
        return segments

    def find_long_enough_segments(self, segments, n=3):
        return [(a, b) for a, b in segments if b - a >= n]
    

    def save_segments_to_raven_file(self,raven_labels,filename,audio_file_name,audio_file_processor):

        self.write_raven_file(raven_labels,filename)


    def find_candidate_raven_files(self,root_path):
        pattern = root_path + '/**/*.txt'
        txtfiles = glob.glob(pattern,recursive=True)
        raven_files = []
        for txtfile in txtfiles:
          tbl = self.ddb.read_csv(txtfile, delimiter='\t', header=True)
          if 'Selection' in tbl.columns:
            raven_files.append(txtfile)
        return raven_files
    
    def load_one_raven_file(self, raven_file):
        """ usage
            rfh.load_one_raven_file('/tmp/rf.raven')
        """

        self.ddb.sql("""
                    create or replace temp view empty_table_all_columns as
                     select * from all_raven_files limit 0 
                     """)
        rf = self.ddb.sql(
            f"""
            CREATE OR REPLACE VIEW one_raven_file AS
            SELECT * FROM read_csv('{raven_file}', auto_type_candidates = ['BIGINT', 'DOUBLE','VARCHAR'])
            union all by name
            (SELECT * FROM empty_table_all_columns) -- just get all the columns
            """
        )
        useful_cols = self.ddb.sql(
            """
            SELECT
                 "File Offset (s)"::double as start_time,
                 "File Offset (s)"::double - "Begin Time (s)"::double + "End Time (s)"::double as end_time,
                 "Low Freq (Hz)" as low_freq,
                 "High Freq (Hz)" as high_freq,
                 "End Time (s)"::double - "Begin Time (s)"::double as duration,
                 REPLACE("Begin File",'dzan','dz') as audio_file,
                 "Tag 1" as tag1,
                 "Tag 2" as tag2,
                 "Tags" as tags,
                 "Notes" as notes,
                 "Score" as score,
                 raven_filename as raven_file
               FROM one_raven_file
               WHERE end_time is not null
            """
        )
        results = [RavenLabel(*row) for row in useful_cols.fetchall()]
        return results


    def all_raven_files_as_one_table(self, raven_files):
        for idx, f in enumerate(raven_files):
            self.ddb.sql(
                f"""
                CREATE OR REPLACE VIEW raven_file_{idx} as
                SELECT *,'{f}' as raven_filename
                FROM read_csv('{f}', auto_type_candidates = ['BIGINT', 'DOUBLE','VARCHAR']);
                """
            )
        tables_to_union = [
            f"select * from raven_file_{idx}" for idx in range(len(raven_files))
        ]
        union_sql = f"""
            CREATE OR REPLACE TEMPORARY TABLE all_raven_files as
            {" UNION ALL BY NAME ".join(tables_to_union)}
            """
        self.ddb.sql(union_sql)

        self.ddb.sql(
            f"""
            CREATE OR REPLACE TEMP TABLE all_raven_labels AS
              SELECT
                 "File Offset (s)"::double as start_time,
                 "File Offset (s)"::double - "Begin Time (s)"::double + "End Time (s)"::double as end_time,
                 "Low Freq (Hz)" as low_freq,
                 "High Freq (Hz)" as high_freq,
                 "End Time (s)"::double - "Begin Time (s)"::double as duration,
                 REPLACE("Begin File",'dzan','dz') as audio_file,
                 "Tag 1" as tag1,
                 "Tag 2" as tag2,
                 "Tags" as tags,
                 "Notes" as notes,
                 "Score" as score,
                 raven_filename as raven_file
               FROM all_raven_files
               WHERE end_time is not null
            """
        )
        return self.ddb.sql("select * from all_raven_files")

    def write_raven_file(self, labels, output_filename):

        raven_file_columns = [
            "Selection",
            "View",
            "Channel",
            "Begin Time (s)",
            "End Time (s)",
            "Low Freq (Hz)",
            "High Freq (Hz)",
            "Begin Date",
            "Begin Hour",
            "Begin Path",
            "Begin File",
            "File Offset (s)",
            "Date Check",
            "Time Check",
            "Score",
            "Tags",
            "Notes",
            "column17",
            "raven_filename",
            "Analyst",
            "Verification_2",
            "OLD-Selection",
            "Site",
            "Channel-OLD",
            "hour",
            "file date",
            "date(raven)",
            "Tag 1",
            "Tag 2",
            "fileDate",
            "Begin Path - old",
        ]

        with open(output_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(raven_file_columns)
            for idx, row in enumerate(labels):
                #row = RavenLabel(*row)
                data = [
                    idx,
                    "Spectrogram 1", # "View",
                    "01", # "Channel",
                    row.bt, # "Begin Time (s)",
                    row.et,# "End Time (s)",
                    row.lf, # "Low Freq (Hz)",
                    row.hf,# "High Freq (Hz)",
                    "01/01/1970",# "Begin Date",
                    "00",# "Begin Hour",
                    row.audio_file,# "Begin Path",
                    row.audio_file,# "Begin File",
                    row.bt,# "File Offset (s)",
                    "01/01/1970",# "Date Check",
                    "00:00:00.000",# "Time Check",
                    row.score,   # "Score",
                    "Maybe Rumble",  # tags
                    "Maybe Rumble",  # Notes
                    "Maybe Rumble",  # column 17
                    "possible_rumbles.txt",
                    "",  # analyst
                    "",  # 'Verification_2',
                    "",  # 'OLD-Selection',
                    "",  # site
                    "",  # channel-old
                    "",  # hour
                    "",  # fileDate
                    "",  # date(raven)
                    "Maybe Rumble",  # tag1
                    "Maybe Rumble",  # tag2
                    "", # "fileDate",
                    "", # "Begin Path - old",
                ]
                writer.writerow(data)

    def get_all_labels_for_wav_file(self,wav_file):
        """
           Usage:
                lbls = rfh.get_all_labels_for_wav_file('CEB1_20111010_000000.wav')
                for idx,row in enumerate(lbls):
                    print(row)
                    if idx>3:
                        break
        
        """
        wav_file_without_path = os.path.basename(wav_file)
        wav_file_pattern = re.sub(r'dz(an)?', 'dz%', wav_file_without_path)
        rs = self.ddb.sql(f"""
            SELECT * From all_raven_labels
            WHERE audio_file ilike '%{wav_file_pattern}%'
            ORDER BY start_time
        """)
        results = [RavenLabel(*row) for row in rs.fetchall()]
        return results
    
if this_should_be_a_unit_test:=False:
    rfh = RavenFileHelper(root_path='/home/ron/proj/elephantlistening/data/Rumble')
    candidate_files = rfh.find_candidate_raven_files('/home/ron/proj/elephantlistening/data/Rumble')
    all_raven_files = rfh.all_raven_files_as_one_table(candidate_files)
    print(all_raven_files.columns)
    lbls = rfh.get_all_labels_for_wav_file('CEB1_20111010_000000.wav')
    for idx,row in enumerate(lbls):
        print(row)
        if idx>3:
            break

    labels = [
        RavenLabel(
            bt=27.34,
            et=30.443,
            lf=22.8,
            hf=79.0,
            duration=3.1030000000000015,
            audio_file="CEB1_20111010_000000.wav",
            t1=None,
            t2=None,
            t3=None,
            notes=None,
            score=None,
            ravenfile="/home/ron/proj/elephantlistening/data/Rumble/Training/Clearings/rumble_clearing_00-24hr_56days.txt",
        ),
    ]
    rfh.write_raven_file(labels, "/tmp/rf.raven")
    rfh.load_one_raven_file('/tmp/rf.raven')
