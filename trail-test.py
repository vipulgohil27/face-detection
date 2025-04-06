import datetime
from unittest.mock import Mock
import pytest

from ImageDeduplicator import ImageDeduplicator  # Replace with your actual module

def test_trial_period_expired():
    app = ImageDeduplicator()
    app.settings = Mock()  # Mock the QSettings object
    past_date = datetime.date.today() - datetime.timedelta(days=30) #set the date to 30 days in the past.
    app.settings.value.return_value = past_date.isoformat()
    app.check_trial_period()

    assert app.trial_ended == True
    assert app.submit_button.isEnabled() == False

def test_trial_period_active():
    app = ImageDeduplicator()
    app.settings = Mock()
    future_date = datetime.date.today() - datetime.timedelta(days=10)
    app.settings.value.return_value = future_date.isoformat()
    app.check_trial_period()

    assert app.trial_ended == False
    assert app.submit_button.isEnabled() == True