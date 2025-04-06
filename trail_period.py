import os
import datetime
import winreg

TRIAL_DAYS = 20
REG_PATH = r"Software\MyAppTrial"  # Change "MyAppTrial" to your app's name


def get_registry_value(name):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_READ) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return value
    except FileNotFoundError:
        return None


def set_registry_value(name, value):
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, REG_PATH) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def check_trial():
    start_date_str = get_registry_value("StartDate")

    if start_date_str is None:
        # First-time run: Set the trial start date
        start_date = datetime.date.today()
        set_registry_value("StartDate", start_date.isoformat())
        return True  # Allow access

    # Convert stored date to datetime object
    start_date = datetime.date.fromisoformat(start_date_str)
    days_used = (datetime.date.today() - start_date).days

    if days_used > TRIAL_DAYS:
        print("Trial period expired! Please purchase a license.")
        return False  # Deny access
    else:
        remaining_days = TRIAL_DAYS - days_used
        print(f"Trial active. {remaining_days} days remaining.")
        return True  # Allow access


# Run the trial check before allowing app execution
if check_trial():
    print("Launching app...")
    # Add your application logic here
else:
    print("Exiting app.")
