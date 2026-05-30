/**
 * Global UI Store - Common UI state
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { HealthResponse } from '@/types/api';

interface UIState {
  // Sidebar
  sidebarOpen: boolean;
  sidebarCollapsed: boolean;

  // Health
  health: HealthResponse | null;
  healthError: string | null;

  // Notifications
  notifications: Notification[];
  unreadCount: number;

  // Theme
  theme: 'light' | 'dark' | 'system';

  // Language
  language: 'en' | 'zh';
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: number;
  read: boolean;
}

interface UIActions {
  // Sidebar
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;

  // Health
  setHealth: (health: HealthResponse) => void;
  setHealthError: (error: string | null) => void;

  // Notifications
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  markNotificationRead: (id: string) => void;
  clearNotifications: () => void;

  // Theme
  setTheme: (theme: 'light' | 'dark' | 'system') => void;

  // Language
  setLanguage: (language: 'en' | 'zh') => void;
}

const initialState: UIState = {
  sidebarOpen: true,
  sidebarCollapsed: false,
  health: null,
  healthError: null,
  notifications: [],
  unreadCount: 0,
  theme: 'system',
  language: 'en',
};

export const useUIStore = create<UIState & UIActions>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Sidebar actions
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),

      // Health actions
      setHealth: (health) => set({ health }),
      setHealthError: (healthError) => set({ healthError }),

      // Notification actions
      addNotification: (notification) => {
        const newNotification: Notification = {
          ...notification,
          id: `notif-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: Date.now(),
          read: false,
        };
        set((state) => ({
          notifications: [newNotification, ...state.notifications].slice(0, 100),
          unreadCount: state.unreadCount + 1,
        }));
      },

      markNotificationRead: (id) => {
        set((state) => {
          const notifications = state.notifications.map((n) =>
            n.id === id ? { ...n, read: true } : n
          );
          const unreadCount = notifications.filter((n) => !n.read).length;
          return { notifications, unreadCount };
        });
      },

      clearNotifications: () => set({ notifications: [], unreadCount: 0 }),

      // Theme actions
      setTheme: (theme) => set({ theme }),

      // Language actions
      setLanguage: (language) => set({ language }),
    }),
    {
      name: 'qlib1-ui',
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        sidebarCollapsed: state.sidebarCollapsed,
        theme: state.theme,
        language: state.language,
      }),
    }
  )
);