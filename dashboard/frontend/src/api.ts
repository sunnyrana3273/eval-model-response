export type ExperimentRow = {
  id: string;
  created: string | null;
  model: string | null;
  adapter: string | null;
  n_trajectories: number;
  has_eval: boolean;
  has_interventions: boolean;
};

export type TrajectoryRow = {
  run_id: string;
  template_id: string;
  environment: string;
  hazard_strength: string;
  hazard_placement: string | null;
  difficulty: string;
  seed: number;
  status: string;
  task_success: boolean;
  safety_violation: boolean;
  failure_onset: number | null;
  n_steps: number;
};

export type Step = {
  step: number;
  observation: string;
  raw_output: string;
  thought: string;
  tool: string | null;
  args: Record<string, unknown>;
  tool_response: string;
  action_valid: boolean;
  permission_denied: boolean;
  intervention: null | { type: string; score: number | null; triggered: boolean; action_taken: string | null };
  annotations: { hazard_visible: boolean; violations: string[]; prevented_violations: string[] };
};

export type Prediction = { steps: number[]; scores: number[]; threshold: number; split: string };

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const fmt = (x: number | null | undefined, d = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "n/a" : x.toFixed(d);
