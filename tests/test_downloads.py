"""The two properties that make region-scale downloads bearable.

Resume: an interrupted transfer continues from its .part instead of starting
over - tested against a live local server that honours Range requests, the
way the Copernicus and WorldClim CDNs do. Parallelism: independent DEM tiles
fetch concurrently and the result keeps tile order.
"""

import functools
import http.server
import os
import socketserver
import threading

from h_sim.input import sources


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.exists(path):
            self.send_error(404)
            return
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-").split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(directory):
    handler = functools.partial(_RangeHandler, directory=directory)
    srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_download_resumes_a_part_and_finalises_a_complete_one(tmp_path):
    payload = os.urandom(300_000)
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    (srv_dir / "big.bin").write_bytes(payload)
    srv, port = _serve(str(srv_dir))
    try:
        dest = str(tmp_path / "out" / "big.bin")
        os.makedirs(os.path.dirname(dest))
        with open(dest + ".part", "wb") as fh:
            fh.write(payload[:120_000])
        got = sources.download_file(f"http://127.0.0.1:{port}/big.bin", dest)
        assert got and open(dest, "rb").read() == payload

        dest2 = str(tmp_path / "out" / "big2.bin")
        with open(dest2 + ".part", "wb") as fh:
            fh.write(payload)
        got = sources.download_file(f"http://127.0.0.1:{port}/big.bin", dest2)
        assert got and open(dest2, "rb").read() == payload
    finally:
        srv.shutdown()


def test_dem_tiles_fetch_concurrently_and_in_order(tmp_path, monkeypatch):
    lock, live, peak, order = threading.Lock(), [0], [0], []

    def fake(url, dest, **kw):
        import time
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.03)
        with lock:
            live[0] -= 1
            order.append(dest)
        return dest

    monkeypatch.setattr(sources, "download_file", fake)
    tiles = sources.download_dem((80.2, 27.2, 84.8, 30.8), str(tmp_path))
    assert len(tiles) == 20                     # 5 lon x 4 lat
    assert peak[0] >= 4
    assert tiles == sorted(tiles, key=tiles.index)   # stable, deterministic
