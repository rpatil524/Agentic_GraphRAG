export interface Entity {
  uid: string;
  name?: string;
  real_label?: string;
  city?: string;
  risk?: number | null;
}

export interface EntityDetails {
  name?: string;
  City?: string;
  Community?: string | number;
  legal_form?: string;
  deletion_date?: string;
  capital_nominal?: string | number;
  capital_paid?: string | number;
  address?: string;
  purpose?: string;
}

export interface PersonRecord {
  Name?: string;
  Origin?: string;
  Since?: string;
  Type?: string;
}

export interface EventRecord {
  ID?: string;
  Date?: string;
  Rubric?: string;
  Text?: string;
}

export interface GraphNode {
  id: string;
  group: number;
  name?: string;
  label?: string;
}

export interface GraphLink {
  source: string;
  target: string;
  type?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  trace?: string;
  records?: Record<string, unknown>[];
}
