/**
 * Backtest Store - Backtest task management
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { BacktestStatus, BacktestRequest, BacktestResult } from '@/types/api';

interface BacktestTask {
  taskId: string;
  request: BacktestRequest;
  status: BacktestStatus;
  submittedAt: string;
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
  getTaskList: () => BacktestTask[];
  deleteTask: (taskId: string) => void;
}

const initialState: BacktestState = {
  tasks: {},
  currentTaskId: null,
  isSubmitting: false,
  error: null,
};

export const useBacktestStore = create<BacktestState & BacktestActions>()(
  persist(
    (set, get) => ({
      ...initialState,

      addTask: (taskId, request, status) => {
        set((state) => ({
          tasks: {
            ...state.tasks,
            [taskId]: {
              taskId,
              request,
              status,
              submittedAt: new Date().toISOString(),
            },
          },
          currentTaskId: taskId,
        }));
      },

      updateTaskStatus: (taskId, status) => {
        set((state) => {
          const existing = state.tasks[taskId];
          if (!existing) return state;
          return {
            tasks: {
              ...state.tasks,
              [taskId]: {
                ...existing,
                status,
              },
            },
          };
        });
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
      getTaskList: () => {
        const { tasks } = get();
        return Object.values(tasks).sort(
          (a, b) => new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime()
        );
      },
      deleteTask: (taskId) => {
        set((state) => {
          const { [taskId]: _, ...rest } = state.tasks;
          return {
            tasks: rest,
            currentTaskId: state.currentTaskId === taskId ? null : state.currentTaskId,
          };
        });
      },
    }),
    {
      name: 'qlib1-backtest-tasks',
      partialize: (state) => ({
        tasks: state.tasks,
        currentTaskId: state.currentTaskId,
      }),
    }
  )
);

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