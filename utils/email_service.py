import os
import smtplib
import logging
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def send_email(to_email: str, subject: str, body: str) -> None:
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_APP_PASSWORD")

    if not sender or not password:
        raise EnvironmentError(
            "EMAIL_SENDER and EMAIL_APP_PASSWORD must be set in .env"
        )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
        server.login(sender, password)
        server.sendmail(sender, to_email, msg.as_string())

    logger.info("Email sent to %s", to_email)