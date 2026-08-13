from unittest.mock import patch, MagicMock


def test_post_call_has_timeout(appointment_nodes):
    make_confirmation_call = appointment_nodes["tools"].make_confirmation_call

    fake_response = MagicMock()
    fake_response.json.return_value = {"status": "ok"}

    with patch(
        "appointment_agent.tools.make_confirmation_call.requests.post",
        return_value=fake_response,
    ) as mock_post:
        make_confirmation_call.invoke(
            {"phone_number": "+15551234567", "instructions": "confirm appointment"}
        )

    _, kwargs = mock_post.call_args
    assert "timeout" in kwargs
    assert kwargs["timeout"] is not None
