import { create } from 'zustand';

interface WSMessage {
  type: string;
  channel?: string;
  data?: unknown;
  timestamp: string;
}

interface WSState {
  connected: boolean;
  messages: WSMessage[];
  lastMessage: WSMessage | null;
}

interface WSActions {
  addMessage: (message: WSMessage) => void;
  setConnected: (connected: boolean) => void;
  clearMessages: () => void;
}

export const useWSStore = create<WSState & WSActions>()((set) => ({
  connected: false,
  messages: [],
  lastMessage: null,

  addMessage: (message) =>
    set((state) => ({
      messages: [message, ...state.messages].slice(0, 200),
      lastMessage: message,
    })),

  setConnected: (connected) => set({ connected }),

  clearMessages: () => set({ messages: [], lastMessage: null }),
}));
