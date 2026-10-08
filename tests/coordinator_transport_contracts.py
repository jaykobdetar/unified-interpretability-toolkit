"""Inert coordinator stream and deadline contracts; no socket or process is opened."""

import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import live_inference as live


class TransportContracts(unittest.TestCase):
    def test_backend_error(self):
        error = live.BackendError(503, "fixture-code", "fixture message")
        self.assertIsInstance(error, Exception)
        self.assertEqual(
            (error.status, error.code, str(error), error.args),
            (503, "fixture-code", "fixture message", ("fixture message",)),
        )

    def test_deadline_init(self):
        clock = mock.Mock(return_value=7.0)
        session = SimpleNamespace(tick=mock.Mock())
        deadline = live.OperationDeadline(session, clock)
        self.assertIs(deadline.clock, clock)
        self.assertIs(deadline.session, session)
        self.assertEqual(deadline.end, 12.0)
        self.assertEqual(deadline.last_tick, -float("inf"))
        clock.assert_called_once_with()
        session.tick.assert_not_called()
        with mock.patch.object(live.time, "monotonic", return_value=9.0) as default:
            second = live.OperationDeadline(session)
        self.assertIs(second.clock, default)
        self.assertEqual(second.end, 14.0)

    def test_remaining(self):
        clock = mock.Mock(return_value=0.0)
        session = SimpleNamespace(tick=mock.Mock())
        deadline = live.OperationDeadline(session, clock)
        with mock.patch.object(live, "available", return_value=6 * live.GIB):
            self.assertEqual(deadline.remaining(), 5.0)
            self.assertEqual(deadline.last_tick, 0.0)
            clock.return_value = 0.05
            self.assertEqual(deadline.remaining(), 4.95)
            self.assertEqual(session.tick.call_count, 1)
            clock.return_value = 5.0
            with self.assertRaises(live.BackendError) as caught:
                deadline.remaining()
            self.assertEqual(
                (caught.exception.status, caught.exception.code, str(caught.exception)),
                (504, "backend_timeout", "Local renderer operation timed out"),
            )
        clock.return_value = 6.0
        with mock.patch.object(live, "available", return_value=3 * live.GIB):
            with self.assertRaises(live.BackendError) as caught:
                deadline.remaining()
        self.assertEqual(
            (caught.exception.status, caught.exception.code, str(caught.exception)),
            (503, "resource_limit", "Local memory reserve reached"),
        )

    def test_reader_init(self):
        connection, deadline = mock.Mock(), mock.Mock()
        reader = live.DeadlineReader(connection, deadline)

        def cleanup():
            reader.connection = connection
            reader.close()

        self.addCleanup(cleanup)
        self.assertIs(reader.connection, connection)
        self.assertIs(reader.deadline, deadline)
        self.assertIs(reader.closed, False)
        connection.close.assert_not_called()

    def test_readable(self):
        reader = live.DeadlineReader(mock.Mock(), mock.Mock())
        self.addCleanup(reader.close)
        self.assertIs(reader.readable(), True)

    def test_readinto(self):
        calls = []

        def receive(buffer):
            calls.append(buffer)
            if len(calls) == 1:
                raise TimeoutError("inert short wait")
            buffer[:2] = b"ab"
            return 2

        connection = SimpleNamespace(
            settimeout=mock.Mock(),
            recv_into=mock.Mock(side_effect=receive),
            close=mock.Mock(),
        )
        deadline = SimpleNamespace(remaining=mock.Mock(return_value=0.07))
        reader = live.DeadlineReader(connection, deadline)
        self.addCleanup(reader.close)
        buffer = bytearray(4)
        self.assertEqual(reader.readinto(buffer), 2)
        self.assertEqual(buffer, b"ab\0\0")
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(value is buffer for value in calls))
        self.assertEqual(
            connection.settimeout.call_args_list, [mock.call(0.07), mock.call(0.07)]
        )
        self.assertEqual(deadline.remaining.call_count, 3)

    def test_reader_close(self):
        connection = SimpleNamespace(close=mock.Mock())
        reader = live.DeadlineReader(connection, mock.Mock())
        self.assertIsNone(reader.close())
        self.assertIs(reader.closed, True)
        connection.close.assert_called_once_with()

    def test_socket_init(self):
        connection, deadline = mock.Mock(), mock.Mock()
        wrapper = live.DeadlineSocket(connection, deadline)
        self.assertIs(wrapper.connection, connection)
        self.assertIs(wrapper.deadline, deadline)
        connection.assert_not_called()
        deadline.assert_not_called()

    def test_sendall(self):
        sent, attempts = bytearray(), []

        def send(data):
            attempts.append(bytes(data))
            if len(attempts) == 1:
                raise TimeoutError("inert short wait")
            size = min(2, len(data))
            sent.extend(data[:size])
            return size

        connection = SimpleNamespace(
            settimeout=mock.Mock(), send=mock.Mock(side_effect=send)
        )
        deadline = SimpleNamespace(remaining=mock.Mock(return_value=0.04))
        error = None
        try:
            result = live.DeadlineSocket(connection, deadline).sendall(b"abcde")
        except (ConnectionError, ValueError, TypeError) as caught:
            error = (type(caught), str(caught))
            result = None
        self.assertIsNone(error)
        self.assertIsNone(result)
        self.assertEqual(sent, b"abcde")
        self.assertEqual(attempts, [b"abcde", b"abcde", b"cde", b"e"])
        self.assertEqual(connection.settimeout.call_args_list, [mock.call(0.04)] * 4)
        self.assertEqual(deadline.remaining.call_count, 5)

    def test_makefile(self):
        duplicate = SimpleNamespace(
            close=mock.Mock(),
            recv_into=mock.Mock(return_value=0),
            settimeout=mock.Mock(),
        )
        connection = SimpleNamespace(
            dup=mock.Mock(return_value=duplicate), close=mock.Mock()
        )
        deadline = SimpleNamespace(remaining=mock.Mock(return_value=0.05))
        wrapper = live.DeadlineSocket(connection, deadline)
        with self.assertRaises(ValueError) as caught:
            wrapper.makefile("wb")
        self.assertEqual(
            str(caught.exception), "Only upstream response reads are supported"
        )
        connection.dup.assert_not_called()
        stream = wrapper.makefile("rb")
        self.addCleanup(stream.close)
        self.assertIsInstance(stream, io.BufferedReader)
        self.assertIsInstance(stream.raw, live.DeadlineReader)
        self.assertIs(stream.raw.connection, duplicate)
        self.assertIs(stream.raw.deadline, deadline)
        connection.dup.assert_called_once_with()
        connection.close.assert_not_called()

    def test_socket_close(self):
        connection = SimpleNamespace(close=mock.Mock(return_value="ignored-result"))
        wrapper = live.DeadlineSocket(connection, mock.Mock())
        self.assertIsNone(wrapper.close())
        connection.close.assert_called_once_with()

    def test_proxy_request(self):
        for method, headers in (("GET", {}), ("POST", {"X-Atlas-Local": "1"})):
            with self.subTest(method=method):
                response = SimpleNamespace(
                    status=207,
                    read=mock.Mock(return_value=b"payload"),
                    getheader=mock.Mock(side_effect=lambda name, default: default),
                    close=mock.Mock(),
                )
                original_socket = object()
                connection = SimpleNamespace(
                    sock=original_socket,
                    connect=mock.Mock(),
                    request=mock.Mock(),
                    getresponse=mock.Mock(return_value=response),
                    close=mock.Mock(),
                )
                deadline = SimpleNamespace(remaining=mock.Mock(return_value=0.1))
                owner = SimpleNamespace(tick=mock.Mock())
                with (
                    mock.patch.object(
                        live, "OperationDeadline", return_value=deadline
                    ) as create_deadline,
                    mock.patch.object(
                        live.http.client, "HTTPConnection", return_value=connection
                    ) as create_connection,
                ):
                    result = live.proxy_request(8797, method, "/api/model", owner)
                self.assertEqual(result, (207, b"payload", "application/octet-stream"))
                create_deadline.assert_called_once_with(owner)
                create_connection.assert_called_once_with(
                    "127.0.0.1", 8797, timeout=0.5
                )
                connection.connect.assert_called_once_with()
                self.assertIsInstance(connection.sock, live.DeadlineSocket)
                self.assertIs(connection.sock.connection, original_socket)
                self.assertIs(connection.sock.deadline, deadline)
                connection.request.assert_called_once_with(
                    method, "/api/model", headers=headers
                )
                response.read.assert_called_once_with(2097153)
                response.getheader.assert_called_once_with(
                    "Content-Type", "application/octet-stream"
                )
                self.assertEqual(deadline.remaining.call_count, 3)
                response.close.assert_called_once_with()
                connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
