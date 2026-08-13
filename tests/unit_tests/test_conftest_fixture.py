def test_appointment_nodes_fixture_avoids_network(appointment_nodes):
    tools_mod = appointment_nodes["tools"]
    names = {t.name for t in tools_mod.schedule_tools_set}
    assert names == {
        "GOOGLECALENDAR_FIND_FREE_SLOTS",
        "GOOGLECALENDAR_CREATE_EVENT",
        "GMAIL_CREATE_EMAIL_DRAFT",
    }
