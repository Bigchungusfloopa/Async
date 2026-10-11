-- Run this once in the Supabase SQL editor (Project -> SQL Editor -> New query).
-- Source: shared plan section 6, plus permissive anon policies and the two private Storage buckets.

create extension if not exists vector;

create table if not exists portfolios (
  id uuid primary key default gen_random_uuid(),
  session_name text, name text, risk_level text, language text default 'en',
  created_at timestamptz default now()
);
create table if not exists holdings (
  id bigserial primary key,
  portfolio_id uuid references portfolios on delete cascade,
  ticker text, amount numeric, quantity numeric, weight numeric
);
create table if not exists price_cache (
  ticker text, date date, adj_close double precision,
  primary key (ticker, date)
);
create table if not exists analysis_runs (
  id bigserial primary key,
  portfolio_id uuid references portfolios on delete cascade,
  risk_label text, metrics jsonb, weights jsonb,
  created_at timestamptz default now()
);
create table if not exists news_insights (
  id bigserial primary key,
  ticker text, lang text, sentiment text, summary text, risks jsonb,
  created_at timestamptz default now()
);
create table if not exists llm_cache (
  key text primary key,          -- sha256(task + prompt + lang)
  task text, lang text, response text,
  provider text, created_at timestamptz default now()
);
-- optional (pgvector brownie): dimension MUST match the embedding model used
create table if not exists news_embeddings (
  id bigserial primary key, ticker text, headline text,
  embedding vector(768)
);

create index if not exists holdings_portfolio_idx on holdings (portfolio_id);
create index if not exists portfolios_session_idx on portfolios (session_name);

-- Hackathon simplicity: no auth. Permissive RLS so the anon key can read/write; session_name replaces login.
do $$
declare t text;
begin
  foreach t in array array['portfolios','holdings','price_cache','analysis_runs','news_insights','llm_cache','news_embeddings']
  loop
    execute format('alter table %I enable row level security', t);
    execute format('drop policy if exists "anon_all" on %I', t);
    execute format('create policy "anon_all" on %I for all to anon using (true) with check (true)', t);
  end loop;
end $$;

-- Private Storage buckets + anon access to objects in them (the models are not secret; the bucket is not public).
insert into storage.buckets (id, name, public) values ('models', 'models', false) on conflict do nothing;
insert into storage.buckets (id, name, public) values ('reports', 'reports', false) on conflict do nothing;

drop policy if exists "anon_models_reports_all" on storage.objects;
create policy "anon_models_reports_all" on storage.objects
  for all to anon
  using (bucket_id in ('models', 'reports'))
  with check (bucket_id in ('models', 'reports'));
