"""
Automated Test Suite for Authentication and Identity Verification
-----------------------------------------------------------------
Comprehensive testing covering:
  1. Registration with strict email format & disposable domain filtering.
  2. Prevention of fake / unverified email registration.
  3. Email verification via 6-digit OTP code.
  4. Rejection of invalid, mismatched, and expired verification codes.
  5. Enforcement of email verification before Sign In (unverified accounts blocked with 403).
  6. Sign In via verified email and phone number.
  7. Forgot password identity verification via registered email and phone.
  8. Password reset with OTP code verification and complexity rules.
  9. Invalidation of old password upon reset.
  10. Session management, authenticated user profile (/api/auth/me), and Logout.
  11. Verification that passwords, OTPs, and sensitive hashes are never leaked in API responses.
"""

import time
import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from backend.app import app
from backend.auth import (
    get_latest_otp_for_test,
    get_db_connection,
    hash_otp,
    _TEST_OTP_STORE,
    init_db
)

@pytest.fixture(scope="module")
def client():
    init_db()
    return TestClient(app)

@pytest.fixture(autouse=True)
def clean_db():
    """Ensure database is clean before running tests."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE email LIKE '%@aviation-safety-test.org' OR email LIKE '%@test-aviation.com'")
    conn.commit()
    conn.close()
    _TEST_OTP_STORE.clear()


# =============================================================================
# 1. Registration & Email Validation Tests
# =============================================================================
class TestRegistrationAndEmailValidation:
    def test_register_valid_user_success(self, client):
        """Valid registration creates account with is_verified=False and sends OTP."""
        payload = {
            "name": "Captain Meera Joshi",
            "email": "meera.joshi@aviation-safety-test.org",
            "phone": "+13125550199",
            "password": "FlightSafety2024!"
        }
        res = client.post("/api/auth/register", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["status"] == "success"
        assert data["is_verified"] is False
        assert "password" not in data
        assert "otp" not in data
        assert "token" not in data  # No token until verified

    def test_register_rejects_disposable_email(self, client):
        """Disposable / fake temporary email domains must be rejected."""
        disposable_emails = [
            "baduser@tempmail.com",
            "spammer@mailinator.com",
            "tester@10minutemail.com",
            "hacker@throwawaymail.com",
            "anon@guerrillamail.com",
        ]
        for bad_email in disposable_emails:
            payload = {
                "name": "Fake User",
                "email": bad_email,
                "password": "Password123"
            }
            res = client.post("/api/auth/register", json=payload)
            assert res.status_code == 400
            assert "disposable" in res.json()["detail"].lower() or "temporary" in res.json()["detail"].lower()

    def test_register_rejects_malformed_email(self, client):
        """Malformed email syntax must be rejected."""
        bad_formats = [
            "not-an-email",
            "missing-domain@",
            "@missing-local.com",
            "user@nodot",
            "user@domain.c",  # TLD < 2 chars
        ]
        for bad_email in bad_formats:
            payload = {
                "name": "Malformed User",
                "email": bad_email,
                "password": "Password123"
            }
            res = client.post("/api/auth/register", json=payload)
            assert res.status_code == 400

    def test_register_enforces_password_strength(self, client):
        """Passwords < 8 chars or missing numbers/letters must be rejected."""
        weak_passwords = [
            "short1",        # < 8 chars
            "alllettersonly",  # No numbers
            "1234567890",      # No letters
        ]
        for weak_pwd in weak_passwords:
            payload = {
                "name": "Weak Pwd User",
                "email": "weak@aviation-safety-test.org",
                "password": weak_pwd
            }
            res = client.post("/api/auth/register", json=payload)
            assert res.status_code in [400, 422]
            assert "password" in str(res.json()["detail"]).lower()

    def test_register_duplicate_email_rejected(self, client):
        """Attempting to register with an already registered email must return 409 Conflict."""
        payload = {
            "name": "Original User",
            "email": "duplicate@aviation-safety-test.org",
            "password": "SecurePassword1"
        }
        r1 = client.post("/api/auth/register", json=payload)
        assert r1.status_code == 201

        r2 = client.post("/api/auth/register", json=payload)
        assert r2.status_code == 409
        assert "already registered" in r2.json()["detail"].lower()


# =============================================================================
# 2. Email Verification Tests
# =============================================================================
class TestEmailVerification:
    def test_verify_email_success(self, client):
        """Entering correct 6-digit OTP verifies account and activates session."""
        email = "verify.me@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Verification Tester",
            "email": email,
            "password": "SecurePassword2024!"
        })

        # Retrieve OTP from internal test store
        otp = get_latest_otp_for_test(email)
        assert otp is not None
        assert len(otp) == 6

        res = client.post("/api/auth/verify-email", json={"email": email, "code": otp})
        assert res.status_code == 200
        data = res.json()
        assert data["is_verified"] is True
        assert "token" in data
        assert data["user"]["email"] == email

    def test_verify_email_with_invalid_code(self, client):
        """Entering an incorrect code returns 400 Bad Request with proper message."""
        email = "wrong.code@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Wrong Code Tester",
            "email": email,
            "password": "SecurePassword2024!"
        })

        res = client.post("/api/auth/verify-email", json={"email": email, "code": "000000"})
        assert res.status_code == 400
        assert "invalid" in res.json()["detail"].lower()

    def test_verify_email_with_expired_code(self, client):
        """Expired verification codes must be rejected with an expiration error."""
        email = "expired@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Expired Tester",
            "email": email,
            "password": "SecurePassword2024!"
        })
        otp = get_latest_otp_for_test(email)

        # Force expiry in the database
        past_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        conn = get_db_connection()
        conn.execute("UPDATE users SET email_otp_expires_at = ? WHERE email = ?", (past_time, email))
        conn.commit()
        conn.close()

        res = client.post("/api/auth/verify-email", json={"email": email, "code": otp})
        assert res.status_code == 400
        assert "expired" in res.json()["detail"].lower()

    def test_resend_verification_code(self, client):
        """Resend verification issues a fresh 6-digit code."""
        email = "resend@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Resend Tester",
            "email": email,
            "password": "SecurePassword2024!"
        })
        otp1 = get_latest_otp_for_test(email)

        res = client.post("/api/auth/resend-verification", json={"email": email})
        assert res.status_code == 200
        assert "new verification code" in res.json()["message"].lower()

        otp2 = get_latest_otp_for_test(email)
        assert otp2 is not None
        # Verify with new code succeeds
        r_ver = client.post("/api/auth/verify-email", json={"email": email, "code": otp2})
        assert r_ver.status_code == 200


# =============================================================================
# 3. Sign In & Access Enforcement Tests
# =============================================================================
class TestSignInAndAccessEnforcement:
    def test_login_unverified_account_blocked(self, client):
        """CRITICAL: User cannot sign in before completing email verification (403 Forbidden)."""
        email = "unverified.login@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Unverified User",
            "email": email,
            "password": "MySecretPassword123"
        })

        # Attempt sign in
        res = client.post("/api/auth/login", json={"identifier": email, "password": "MySecretPassword123"})
        assert res.status_code == 403
        assert "not verified" in res.json()["detail"].lower()

    def test_login_verified_account_success(self, client):
        """Verified user successfully signs in and receives a session token."""
        email = "verified.login@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Verified Pilot",
            "email": email,
            "phone": "+13125550200",
            "password": "AviationPilot99!"
        })
        otp = get_latest_otp_for_test(email)
        client.post("/api/auth/verify-email", json={"email": email, "code": otp})

        # Sign in with email
        r_email = client.post("/api/auth/login", json={"identifier": email, "password": "AviationPilot99!"})
        assert r_email.status_code == 200
        assert "token" in r_email.json()
        assert r_email.json()["user"]["is_verified"] is True

        # Sign in with phone number
        r_phone = client.post("/api/auth/login", json={"identifier": "+13125550200", "password": "AviationPilot99!"})
        assert r_phone.status_code == 200
        assert "token" in r_phone.json()

    def test_login_invalid_password_rejected(self, client):
        """Sign in with incorrect password returns 401 Unauthorized."""
        email = "bad.pass@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Wrong Pass User",
            "email": email,
            "password": "CorrectPassword123"
        })
        otp = get_latest_otp_for_test(email)
        client.post("/api/auth/verify-email", json={"email": email, "code": otp})

        res = client.post("/api/auth/login", json={"identifier": email, "password": "WrongPassword123"})
        assert res.status_code == 401
        assert "invalid" in res.json()["detail"].lower()


# =============================================================================
# 4. Forgot Password & Identity Verification Tests
# =============================================================================
class TestForgotPasswordAndReset:
    def test_forgot_password_via_email(self, client):
        """Requesting password reset via registered email generates an OTP."""
        email = "reset.email@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Email Reset User",
            "email": email,
            "password": "OldPassword123"
        })
        otp = get_latest_otp_for_test(email)
        client.post("/api/auth/verify-email", json={"email": email, "code": otp})

        res = client.post("/api/auth/forgot-password", json={"identifier": email})
        assert res.status_code == 200
        assert res.json()["delivery_channel"] == "email"

        reset_code = get_latest_otp_for_test(email)
        assert reset_code is not None
        assert len(reset_code) == 6

    def test_forgot_password_via_phone(self, client):
        """Requesting password reset via registered phone verifies identity and dispatches code."""
        phone = "+13125550300"
        email = "phone.reset@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Phone Reset User",
            "email": email,
            "phone": phone,
            "password": "OldPassword123"
        })
        otp = get_latest_otp_for_test(email)
        client.post("/api/auth/verify-email", json={"email": email, "code": otp})

        res = client.post("/api/auth/forgot-password", json={"identifier": phone})
        assert res.status_code == 200
        assert res.json()["delivery_channel"] == "phone"

    def test_forgot_password_nonexistent_user(self, client):
        """Unknown email/phone returns 404 Not Found."""
        res = client.post("/api/auth/forgot-password", json={"identifier": "nonexistent@aviation-safety-test.org"})
        assert res.status_code == 404

    def test_reset_password_with_valid_code(self, client):
        """Providing valid reset code updates the password and enables sign in with new credentials."""
        email = "complete.reset@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Complete Reset User",
            "email": email,
            "password": "OriginalPassword1"
        })
        otp = get_latest_otp_for_test(email)
        client.post("/api/auth/verify-email", json={"email": email, "code": otp})

        # Request reset
        client.post("/api/auth/forgot-password", json={"identifier": email})
        reset_code = get_latest_otp_for_test(email)

        # Reset password
        res = client.post("/api/auth/reset-password", json={
            "identifier": email,
            "code": reset_code,
            "new_password": "NewUpdatedPassword2!"
        })
        assert res.status_code == 200
        assert "reset successfully" in res.json()["message"].lower()

        # Sign in with old password fails
        r_old = client.post("/api/auth/login", json={"identifier": email, "password": "OriginalPassword1"})
        assert r_old.status_code == 401

        # Sign in with new password succeeds
        r_new = client.post("/api/auth/login", json={"identifier": email, "password": "NewUpdatedPassword2!"})
        assert r_new.status_code == 200

    def test_reset_password_with_invalid_code(self, client):
        """Reset password with wrong code fails with 400 Bad Request."""
        email = "bad.code.reset@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Bad Reset Code",
            "email": email,
            "password": "OriginalPassword1"
        })
        otp = get_latest_otp_for_test(email)
        client.post("/api/auth/verify-email", json={"email": email, "code": otp})
        client.post("/api/auth/forgot-password", json={"identifier": email})

        res = client.post("/api/auth/reset-password", json={
            "identifier": email,
            "code": "999999",  # Invalid code
            "new_password": "NewUpdatedPassword2!"
        })
        assert res.status_code == 400
        assert "invalid" in res.json()["detail"].lower()


# =============================================================================
# 5. Session Profile & Logout Tests
# =============================================================================
class TestSessionManagementAndLogout:
    def test_authenticated_profile_and_logout(self, client):
        """Test GET /api/auth/me returns profile and POST /api/auth/logout revokes session."""
        email = "session.user@aviation-safety-test.org"
        client.post("/api/auth/register", json={
            "name": "Session Officer",
            "email": email,
            "password": "OfficerPassword123"
        })
        otp = get_latest_otp_for_test(email)
        v_res = client.post("/api/auth/verify-email", json={"email": email, "code": otp})
        token = v_res.json()["token"]

        headers = {"Authorization": f"Bearer {token}"}

        # Profile fetch
        me_res = client.get("/api/auth/me", headers=headers)
        assert me_res.status_code == 200
        assert me_res.json()["user"]["name"] == "Session Officer"

        # Logout
        logout_res = client.post("/api/auth/logout", headers=headers)
        assert logout_res.status_code == 200

        # Attempt to access profile after logout returns 401
        me_after = client.get("/api/auth/me", headers=headers)
        assert me_after.status_code == 401
