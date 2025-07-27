from datetime import datetime
import shutil

def BackupFile(OutputPath: str):
    """
    Updates the name of an existing file to include current time and date for backup.

    :param OutputPath: Name of the file to be backed up.
    """
    Timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BackupPath = OutputPath.replace(".csv", f"_backup_{Timestamp}.csv")
    shutil.move(OutputPath, BackupPath)
    print(f"Backed up {OutputPath} to: {BackupPath}")