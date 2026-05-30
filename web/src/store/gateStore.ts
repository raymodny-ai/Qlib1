/**
 * Gate Store - PM Gate status management
 */

import { create } from 'zustand';
import type { GateStatus as GateStatusType, GateHistoryEntry, GateDimension, GateState as GateStateType } from '@/types/api';

interface GateStoreState {
  status: GateStatusType | null;
  history: GateHistoryEntry[];
  isLoading: boolean;
  isPolling: boolean;
  error: string | null;
}

interface GateStoreActions {
  setStatus: (status: GateStatusType) => void;
  setHistory: (history: GateHistoryEntry[]) => void;
  setLoading: (loading: boolean) => void;
  setPolling: (polling: boolean) => void;
  setError: (error: string | null) => void;
  updateGateState: (dimension: GateDimension, newState: GateStateType) => void;
  clearError: () => void;
}

const initialState: GateStoreState = {
  status: null,
  history: [],
  isLoading: false,
  isPolling: false,
  error: null,
};

export const useGateStore = create<GateStoreState & GateStoreActions>()((set, get) => ({
  ...initialState,

  setStatus: (status) => set({ status }),
  setHistory: (history) => set({ history }),
  setLoading: (isLoading) => set({ isLoading }),
  setPolling: (isPolling) => set({ isPolling }),
  setError: (error) => set({ error }),

  updateGateState: (dimension: GateDimension, newState: GateStateType) => {
    const { status } = get();
    if (!status) return;

    set({
      status: {
        ...status,
        gates: {
          ...status.gates,
          [dimension]: newState,
        },
        [`can_${getGateAction(dimension)}`]: newState === 'open',
        is_any_closed: Object.values({
          ...status.gates,
          [dimension]: newState,
        }).includes('closed'),
      },
    });
  },

  clearError: () => set({ error: null }),
}));

// Helper to get action name from dimension
function getGateAction(dimension: GateDimension): string {
  const actionMap: Record<GateDimension, string> = {
    signal: 'push_signal',
    train: 'train_model',
    deploy: 'deploy_model',
  };
  return actionMap[dimension];
}

// Gate dimension labels
export const gateDimensionLabels: Record<GateDimension, string> = {
  signal: 'Signal Push',
  train: 'Model Training',
  deploy: 'Model Deploy',
};

export const gateDimensionDescriptions: Record<GateDimension, string> = {
  signal: 'Controls whether trading signals are pushed to downstream systems',
  train: 'Controls whether new model training tasks can be started',
  deploy: 'Controls whether models can be deployed to production',
};