from datetime import datetime
from typing import *
import csv


def now():
    return datetime.now()


def save_csv(data: List[Tuple[Any]], path: str):
    with open(path, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)

    csvFile.close()
