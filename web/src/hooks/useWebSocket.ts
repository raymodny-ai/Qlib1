import { useEffect, useRef, useCallback } from 'react';
import { useWSStore } from '@/store/wsStore';

type WSChannel = 'gate' | 'backtest' | 'system' | 'alerts';

interface UseWebSocketOptions {
  channel?: WSChannel;
  onMessage?: (data: unknown) => void;
  enabled?: boolean;
}

export function useWebSocket({ channel, onMessage, enabled = true }: UseWebSocketOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);
  const { addMessage, setConnected } = useWSStore();

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/monitor`;

    // Pass auth token if available
    const token = localStorage.getItem('qlib1_token');
    const url = token ? `${wsUrl}?token=${token}` : wsUrl;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (channel) {
        ws.send(JSON.stringify({ action: 'subscribe', channel }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        addMessage(data);
        onMessage?.(data);
      } catch {
        // Non-JSON message, ignore
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      // Auto-reconnect after 5 seconds
      if (enabled) {
        setTimeout(connect, 5000);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [channel, onMessage, enabled, addMessage, setConnected]);

  useEffect(() => {
    if (!enabled) return;
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect on unmount
        wsRef.current.close();
      }
    };
  }, [connect, enabled]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { send };
}
