/**
 * Backtest Store - Backtest task management
 */

import { create } from 'zustand';
import type { BacktestStatus, BacktestRequest, BacktestResult } from '@/types/api';

interface BacktestTask {
  taskId: string;
  request: BacktestRequest;
  status: BacktestStatus;
}

interface BacktestState {
  tasks: Record<string, BacktestTask>;
  currentTaskId: string | null;
  isSubmitting: boolean;
  error: string | null;
}

interface BacktestActions {
  addTask: (taskId: string, request: BacktestRequest, status: BacktestStatus) => void;
  updateTaskStatus: (taskId: string, status: BacktestStatus) => void;
  setCurrentTask: (taskId: string | null) => void;
  setSubmitting: (submitting: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  getTask: (taskId: string) => BacktestTask | undefined;
  getCurrentTask: () => BacktestTask | undefined;
}

const initialState: BacktestState = {
  tasks: {},
  currentTaskId: null,
  isSubmitting: false,
  error: null,
};

export const useBacktestStore = create<BacktestState & BacktestActions>()((set, get) => ({
  ...initialState,

  addTask: (taskId, request, status) => {
    set((state) => ({
      tasks: {
        ...state.tasks,
        [taskId]: {
          taskId,
          request,
          status,
        },
      },
      currentTaskId: taskId,
    }));
  },

  updateTaskStatus: (taskId, status) => {
    set((state) => ({
      tasks: {
        ...state.tasks,
        [taskId]: {
          ...state.tasks[taskId],
          status,
        },
      },
    }));
  },

  setCurrentTask: (taskId) => set({ currentTaskId: taskId }),
  setSubmitting: (isSubmitting) => set({ isSubmitting }),
  setError: (error) => set({ error }),
  clearError: () => set({ error: null }),

  getTask: (taskId) => get().tasks[taskId],
  getCurrentTask: () => {
    const { tasks, currentTaskId } = get();
    return currentTaskId ? tasks[currentTaskId] : undefined;
  },
}));

// Backtest status colors for UI
export const backtestStatusColors: Record<BacktestStatus['status'], string> = {
  pending: '#f59e0b',
  running: '#3b82f6',
  completed: '#22c55e',
  failed: '#ef4444',
};

// Backtest status labels
export const backtestStatusLabels: Record<BacktestStatus['status'], string> = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
};