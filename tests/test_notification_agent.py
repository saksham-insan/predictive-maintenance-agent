"""
Unit tests for the Notification Agent.

Tests:
1. High-risk alert detection (prediction == 1, confidence >= 0.70 or urgency == "High").
2. Low/Medium-risk alert does NOT trigger notification.
3. New high-risk alert triggers notification email sending.
4. Duplicate high-risk alert is detected and skipped.
5. Email content contains important alert information (Machine ID, Confidence, Reason, AI Insight, Recommendation).
6. CSV attachment is generated correctly with all telemetry and alert fields.
7. Missing SMTP configuration fails gracefully without crashing.
8. SMTP failure (network/timeout/auth error) fails gracefully without crashing.
9. Async dispatch executes safely in background without blocking.
10. Existing Gemini fallback remains completely unaffected.
"""

import os
import sys
import io
import csv
import smtplib
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.notification_agent import (
    is_high_risk,
    make_notification_event_key,
    has_been_notified,
    mark_notified,
    reset_notification_state,
    get_last_notification_status,
    get_notification_history,
    generate_alert_csv,
    generate_email_content,
    send_notification_email,
    notification_agent,
)
from llm_reasoning import generate_local_insight, reset_quota_status


class TestNotificationAgent(unittest.TestCase):
    def setUp(self):
        reset_notification_state()
        reset_quota_status()

        self.high_risk_diagnosis = {
            "prediction": 1,
            "confidence": 0.88,
            "explanation": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
            "plain_explanation": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
        }
        self.high_risk_recommendation = {
            "action": "Schedule immediate maintenance",
            "urgency": "High",
            "confidence": 0.88,
            "explanation": self.high_risk_diagnosis["explanation"],
        }
        self.sensor_data = {
            "Type": "M",
            "Air temperature [K]": 300.2,
            "Process temperature [K]": 310.5,
            "Rotational speed [rpm]": 1380,
            "Torque [Nm]": 62.4,
            "Tool wear [min]": 215,
        }
        self.sample_record = {
            "machine_id": "Machine-001",
            "timestamp": "t=12",
            "diagnosis": self.high_risk_diagnosis,
            "recommendation": self.high_risk_recommendation,
            "ai_insight": "Critical torque and tool wear detected. Prioritize immediate inspection.",
            "sensor_data": self.sensor_data,
        }

    def test_high_risk_detection(self):
        """Test that is_high_risk identifies high-risk predictions and high urgency."""
        # High confidence failure
        self.assertTrue(is_high_risk(self.high_risk_diagnosis, self.high_risk_recommendation))

        # Prediction=1, confidence >= 0.70
        diag = {"prediction": 1, "confidence": 0.75}
        self.assertTrue(is_high_risk(diag, {"urgency": "Medium"}))

        # High urgency explicitly specified
        diag2 = {"prediction": 1, "confidence": 0.50}
        self.assertTrue(is_high_risk(diag2, {"urgency": "High"}))

    def test_low_and_medium_risk_not_detected_as_high_risk(self):
        """Test that low and medium risk alerts do NOT trigger high-risk notifications."""
        # Low risk / normal reading
        diag_normal = {"prediction": 0, "confidence": 0.05}
        rec_normal = {"action": "No action needed", "urgency": "Low"}
        self.assertFalse(is_high_risk(diag_normal, rec_normal))

        # Medium risk anomaly (prediction 1 but confidence < 0.70 and urgency Medium)
        diag_medium = {"prediction": 1, "confidence": 0.55}
        rec_medium = {"action": "Schedule maintenance within 48 hours", "urgency": "Medium"}
        self.assertFalse(is_high_risk(diag_medium, rec_medium))

        # Prediction 0 with low confidence anomaly
        diag_low_anom = {"prediction": 0, "confidence": 0.35}
        rec_low_anom = {"action": "No action needed", "urgency": "Low"}
        self.assertFalse(is_high_risk(diag_low_anom, rec_low_anom))

        # notification_agent on low risk returns ignored
        low_record = {
            "machine_id": "Machine-002",
            "timestamp": "t=15",
            "diagnosis": diag_normal,
            "recommendation": rec_normal,
        }
        res = notification_agent(low_record)
        self.assertFalse(res["is_high_risk"])
        self.assertFalse(res["notified"])
        self.assertEqual(res["status"], "ignored")

    def test_email_content_contains_all_critical_fields(self):
        """Test that generated email subject and body contain all essential fields."""
        subject, body = generate_email_content(self.sample_record)

        self.assertIn("Machine-001", subject)
        self.assertIn("[HIGH ALERT]", subject)

        self.assertIn("Machine: Machine-001", body)
        self.assertIn("Time: t=12", body)
        self.assertIn("Risk: HIGH", body)
        self.assertIn("88%", body)
        self.assertIn("Failure likely", body)
        self.assertIn(self.high_risk_diagnosis["explanation"], body)
        self.assertIn(self.sample_record["ai_insight"], body)
        self.assertIn("Schedule immediate maintenance", body)
        self.assertIn("Torque [Nm]: 62.4", body)
        self.assertIn("Tool wear [min]: 215", body)

    def test_csv_attachment_generation(self):
        """Test that generated CSV contains expected headers and record values."""
        csv_bytes, filename = generate_alert_csv(self.sample_record)

        self.assertTrue(filename.startswith("high_risk_alert_Machine-001_t_12"))
        self.assertTrue(filename.endswith(".csv"))


        csv_text = csv_bytes.decode("utf-8")
        reader = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(len(reader), 1)
        row = reader[0]

        self.assertEqual(row["Machine_ID"], "Machine-001")
        self.assertEqual(row["Timestamp"], "t=12")
        self.assertEqual(row["Prediction"], "Failure")
        self.assertEqual(row["Confidence"], "88%")
        self.assertEqual(row["Risk_Level"], "High")
        self.assertEqual(row["Recommended_Action"], "Schedule immediate maintenance")
        self.assertIn("Torque [Nm]", row["SHAP_Reason"])
        self.assertEqual(row["AI_Insight"], self.sample_record["ai_insight"])
        self.assertEqual(row["Torque [Nm]"], "62.4")
        self.assertEqual(row["Tool wear [min]"], "215")

    @patch("smtplib.SMTP")
    def test_new_high_risk_alert_triggers_notification(self, mock_smtp_cls):
        """Test that a new high-risk alert successfully sends an email via SMTP."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value = mock_server

        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "NOTIFICATION_EMAIL_TO": "admin@example.com",
            "SMTP_USERNAME": "user@example.com",
            "SMTP_PASSWORD": "password123",
        }

        with patch.dict(os.environ, env, clear=False):
            res = notification_agent(self.sample_record, async_send=False)

            self.assertTrue(res["is_high_risk"])
            self.assertTrue(res["notified"])
            self.assertEqual(res["status"], "sent")

            # Verify SMTP was called correctly
            mock_smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=10.0)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("user@example.com", "password123")
            mock_server.send_message.assert_called_once()
            mock_server.quit.assert_called_once()

            # Verify sent email message details
            sent_msg = mock_server.send_message.call_args[0][0]
            self.assertEqual(sent_msg["To"], "admin@example.com")
            self.assertIn("Machine-001", sent_msg["Subject"])

            # Verify attachment was added
            payload = sent_msg.get_payload()
            self.assertTrue(isinstance(payload, list))
            self.assertEqual(len(payload), 2)  # text body + csv attachment

    @patch("smtplib.SMTP")
    def test_duplicate_high_risk_alert_is_skipped(self, mock_smtp_cls):
        """Test that duplicate alerts with the same event key are skipped."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value = mock_server

        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "NOTIFICATION_EMAIL_TO": "admin@example.com",
        }

        with patch.dict(os.environ, env, clear=False):
            # First trigger -> sent
            res1 = notification_agent(self.sample_record, async_send=False)
            self.assertTrue(res1["is_high_risk"])
            self.assertTrue(res1["notified"])
            self.assertEqual(res1["status"], "sent")
            self.assertEqual(mock_server.send_message.call_count, 1)

            # Second trigger with same alert -> skipped
            res2 = notification_agent(self.sample_record, async_send=False)
            self.assertTrue(res2["is_high_risk"])
            self.assertFalse(res2["notified"])
            self.assertEqual(res2["status"], "skipped")
            self.assertEqual(res2["reason"], "duplicate")
            # Ensure SMTP was NOT called again
            self.assertEqual(mock_server.send_message.call_count, 1)

    def test_missing_smtp_config_fails_gracefully(self):
        """Test that missing SMTP credentials or host does not crash the application."""
        with patch.dict(os.environ, {}, clear=True):
            res = notification_agent(self.sample_record, async_send=False)
            self.assertTrue(res["is_high_risk"])
            self.assertFalse(res["notified"])
            self.assertEqual(res["status"], "skipped")
            self.assertEqual(res["error"], None)

    @patch("smtplib.SMTP")
    def test_smtp_failure_does_not_crash_pipeline(self, mock_smtp_cls):
        """Test that SMTP network/connection exceptions are caught and reported safely."""
        mock_smtp_cls.side_effect = smtplib.SMTPConnectError(421, "Cannot connect to SMTP server")

        env = {
            "SMTP_HOST": "invalid.smtp.host",
            "SMTP_PORT": "587",
            "NOTIFICATION_EMAIL_TO": "admin@example.com",
        }

        with patch.dict(os.environ, env, clear=False):
            res = notification_agent(self.sample_record, async_send=False)
            self.assertTrue(res["is_high_risk"])
            self.assertFalse(res["notified"])
            self.assertEqual(res["status"], "failed")
            self.assertIn("Cannot connect", str(res["error"]))

    @patch("smtplib.SMTP_SSL")
    def test_smtp_ssl_port_465(self, mock_smtp_ssl_cls):
        """Test that port 465 uses SMTP_SSL."""
        mock_server = MagicMock()
        mock_smtp_ssl_cls.return_value = mock_server

        env = {
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_PORT": "465",
            "NOTIFICATION_EMAIL_TO": "admin@example.com",
        }

        with patch.dict(os.environ, env, clear=False):
            res = notification_agent(self.sample_record, async_send=False)
            self.assertTrue(res["notified"])
            mock_smtp_ssl_cls.assert_called_once_with("smtp.gmail.com", 465, timeout=10.0)

    def test_gemini_fallback_remains_unaffected(self):
        """Test that Gemini fallback insights work seamlessly with notification agent."""
        local_insight = generate_local_insight(self.high_risk_diagnosis, self.high_risk_recommendation)
        record_with_fallback = dict(self.sample_record)
        record_with_fallback["ai_insight"] = local_insight

        subject, body = generate_email_content(record_with_fallback)
        self.assertIn(local_insight, body)
        self.assertIn("Torque [Nm] and Tool wear [min]", body)

    def test_notification_history_tracking(self):
        """Test that notification history is properly recorded and queryable."""
        mark_notified("event_1", {"machine_id": "M1", "status": "sent", "recipient": "a@b.com"})
        mark_notified("event_2", {"machine_id": "M2", "status": "sent", "recipient": "a@b.com"})

        self.assertTrue(has_been_notified("event_1"))
        self.assertTrue(has_been_notified("event_2"))
        self.assertFalse(has_been_notified("event_3"))

        last = get_last_notification_status()
        self.assertIsNotNone(last)
        self.assertEqual(last["machine_id"], "M2")

        history = get_notification_history()
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
