import socket
import threading
import queue
import numpy as np

class AudioServer:
    """Host mode: listends on a UDP port, receives audio, and sends back to caller."""
    def __init__(self, host_ip="0.0.0.0", port=5005):
        self.host_ip = host_ip
        self.port = port
        
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow port reuse just in case
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host_ip, self.port))
        
        # We need to know who to send to. This is capturing the remote client IP.
        self.client_address = None
        
        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        
        self.recv_thread.start()
        self.send_thread.start()
        
    def _receive_loop(self):
        # 1.5 seconds of 16kHz float32 is 24000 samples. Converted to int16, it is 48000 bytes.
        # Max UDP size is 65507 bytes, so it fits in a single datagram.
        buffer_size = 65535
        try:
            while self.running:
                data, addr = self.sock.recvfrom(buffer_size)
                if not data: continue
                
                # First packet sets the target client address for 2-way call
                if self.client_address is None or self.client_address != addr:
                    print(f"Connection established with {addr}")
                    self.client_address = addr
                
                int16_chunk = np.frombuffer(data, dtype=np.int16)
                float32_chunk = int16_chunk.astype(np.float32) / 32767.0
                
                self.receive_queue.put(float32_chunk)
        except Exception as e:
            if self.running: print(f"Server receive error: {e}")
            
    def _send_loop(self):
        try:
            while self.running:
                chunk = self.send_queue.get()
                if chunk is None: break
                
                if self.client_address is None:
                    # Can't send until a client connects and we get their addr
                    continue
                    
                int16_chunk = (chunk * 32767).astype(np.int16)
                data = int16_chunk.tobytes()
                
                self.sock.sendto(data, self.client_address)
        except Exception as e:
            if self.running: print(f"Server send error: {e}")
            
    def stop(self):
        self.running = False
        self.send_queue.put(None)
        self.receive_queue.put(None)
        try:
            self.sock.close()
        except:
            pass


class AudioClient:
    """Join mode: sends audio to a target IP and listens for response on a UDP socket."""
    def __init__(self, target_ip, port=5005):
        self.target_ip = target_ip
        self.port = port
        
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to '0.0.0.0' with any available port to receive responses
        self.sock.bind(('0.0.0.0', 0))
        
        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        
        self.recv_thread.start()
        self.send_thread.start()
        
    def _receive_loop(self):
        buffer_size = 65535
        try:
            while self.running:
                data, addr = self.sock.recvfrom(buffer_size)
                if not data: continue
                
                int16_chunk = np.frombuffer(data, dtype=np.int16)
                float32_chunk = int16_chunk.astype(np.float32) / 32767.0
                
                self.receive_queue.put(float32_chunk)
        except Exception as e:
            if self.running: print(f"Client receive error: {e}")
            
    def _send_loop(self):
        target_addr = (self.target_ip, self.port)
        try:
            while self.running:
                chunk = self.send_queue.get()
                if chunk is None: break
                
                int16_chunk = (chunk * 32767).astype(np.int16)
                data = int16_chunk.tobytes()
                
                self.sock.sendto(data, target_addr)
        except Exception as e:
            if self.running: print(f"Client send error: {e}")
            
    def stop(self):
        self.running = False
        self.send_queue.put(None)
        self.receive_queue.put(None)
        try:
            self.sock.close()
        except:
            pass


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()
