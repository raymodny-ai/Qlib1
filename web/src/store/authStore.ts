/**
 * Auth Store - User authentication and role management
 * Using Zustand for global state management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Role, Permission, ROLE_PERMISSIONS } from '@/types/api';

interface AuthState {
  userId: string;
  role: Role;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

interface AuthActions {
  setUserId: (userId: string) => void;
  setRole: (role: Role) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  login: (userId: string) => void;
  logout: () => void;
  hasPermission: (permission: Permission) => boolean;
}

const rolePermissions: Record<Role, Permission[]> = {
  admin: [
    'model:read',
    'experiment:read',
    'experiment:submit',
    'report:read',
    'signal:emergency_stop',
    'audit:read',
    'audit:export',
    'compliance:export',
    'compliance:review',
    'logs:read',
    'logs:export',
  ],
  researcher: [
    'model:read',
    'experiment:read',
    'experiment:submit',
    'report:read',
    'logs:read',
  ],
  pm: [
    'experiment:read',
    'report:read',
    'signal:emergency_stop',
    'logs:read',
  ],
  auditor: [
    'audit:read',
    'audit:export',
    'compliance:export',
    'compliance:review',
    'logs:read',
    'logs:export',
  ],
};

const initialState: AuthState = {
  userId: 'anonymous',
  role: 'admin',
  isAuthenticated: false,
  isLoading: false,
  error: null,
};

export const useAuthStore = create<AuthState & AuthActions>()(
  persist(
    (set, get) => ({
      ...initialState,

      setUserId: (userId) => set({ userId }),
      setRole: (role) => set({ role }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),

      login: (userId) => {
        const role = mapUserIdToRole(userId);
        set({
          userId,
          role,
          isAuthenticated: true,
          error: null,
        });
        // Also store in localStorage for API header injection
        if (typeof window !== 'undefined') {
          localStorage.setItem('qlib1_user_id', userId);
          localStorage.setItem('qlib1_role', role);
        }
      },

      logout: () => {
        set({ ...initialState });
        if (typeof window !== 'undefined') {
          localStorage.removeItem('qlib1_user_id');
          localStorage.removeItem('qlib1_role');
        }
      },

      hasPermission: (permission) => {
        const { role } = get();
        return rolePermissions[role]?.includes(permission) ?? false;
      },
    }),
    {
      name: 'qlib1-auth',
      partialize: (state) => ({
        userId: state.userId,
        role: state.role,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// Helper function to map user ID to role
function mapUserIdToRole(userId: string): Role {
  const roleMap: Record<string, Role> = {
    admin: 'admin',
    researcher: 'researcher',
    pm: 'pm',
    auditor: 'auditor',
  };

  // Handle portfolio_manager -> pm
  if (userId === 'pm') return 'pm';
  if (userId === 'portfolio_manager') return 'pm';

  return (roleMap[userId] as Role) || 'admin';
}

// Role display names
export const roleLabels: Record<Role, string> = {
  admin: 'System Admin',
  researcher: 'Quant Researcher',
  pm: 'Portfolio Manager',
  auditor: 'Compliance Auditor',
};

// Permission display names
export const permissionLabels: Record<Permission, string> = {
  'model:read': 'Model Prediction',
  'experiment:read': 'View Experiments',
  'experiment:submit': 'Submit Backtest',
  'report:read': 'View Reports',
  'signal:emergency_stop': 'Emergency Stop',
  'audit:read': 'View Audit Logs',
  'audit:export': 'Export Audit',
  'compliance:export': 'Generate SOX Report',
  'compliance:review': 'View Compliance',
  'logs:read': 'View System Logs',
  'logs:export': 'Export System Logs',
};