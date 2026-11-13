import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { FileImage, FileText, ImageUp, X } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { toast } from "sonner";
import { API, api, loadSession } from "../lib/api";
import { MATERIALS } from "../lib/format";
import { useMeta, useSites } from "../lib/queries";
import type { Manuscript } from "../lib/types";
import { fileToTitle } from "./DirectUploadModal";
import { Button, Field, Input, Modal, Progress, Select, Textarea } from "./ui";

const ACCEPT = "image/jpeg,image/png,image/tiff,image/webp,image/bmp,application/pdf";
const MAX_MB = 25;

interface FormState {
  title: string;
  language: string;
  century: string;
  material: string;
  repository: string;
  site_id: string;
  description: string;
  license: string;
  source_url: string;
}

const EMPTY: FormState = {
  title: "",
  language: "sa",
  century: "",
  material: "paper",
  repository: "",
  site_id: "",
  description: "",
  license: "",
  source_url: "",
};

export function ManuscriptForm({
  open,
  onClose,
  initial,
  onSaved,
}: {
  open: boolean;
  onClose: () => void;
  initial?: Manuscript;
  onSaved: (m: Manuscript) => void;
}) {
  const meta = useMeta();
  const sites = useSites();
  const qc = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [form, setForm] = useState<FormState>(EMPTY);
  const [errors, setErrors] = useState<Partial<Record<keyof FormState, string>>>({});
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);

  useEffect(() => {
    if (!open) {
      setFiles([]);
      previews.forEach((u) => URL.revokeObjectURL(u));
      setPreviews([]);
      setProgress(null);
      return;
    }
    setErrors({});
    setForm(
      initial
        ? {
            title: initial.title,
            language: initial.language,
            century: initial.century ? String(initial.century) : "",
            material: initial.material,
            repository: initial.repository,
            site_id: initial.site_id ?? "",
            description: initial.description,
            license: initial.license,
            source_url: initial.source_url,
          }
        : EMPTY,
    );
  }, [open, initial]);

  function addFiles(list: FileList | null) {
    if (!list) return;
    const ok: File[] = [];
    for (const f of Array.from(list)) {
      if (!ACCEPT.split(",").includes(f.type)) {
        toast.error(`${f.name}: only JPG, PNG, TIFF, WebP, BMP or PDF supported`);
        continue;
      }
      if (f.size > MAX_MB * 1024 * 1024) {
        toast.error(`${f.name} is larger than ${MAX_MB} MB`);
        continue;
      }
      ok.push(f);
    }
    if (!ok.length) return;

    setFiles((prev) => {
      const merged = [...prev, ...ok].slice(0, 50);
      if (!form.title.trim() && merged.length > 0) {
        setForm((curr) => ({ ...curr, title: fileToTitle(merged[0].name) }));
      }
      return merged;
    });

    const urls = ok.map((f) => (f.type.startsWith("image/") ? URL.createObjectURL(f) : ""));
    setPreviews((prev) => [...prev, ...urls].slice(0, 50));
  }

  function removeFile(index: number) {
    if (previews[index]) URL.revokeObjectURL(previews[index]);
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => prev.filter((_, i) => i !== index));
  }

  const save = useMutation({
    mutationFn: (body: object) =>
      initial ? api.patch<Manuscript>(`/manuscripts/${initial.id}`, body) : api.post<Manuscript>("/manuscripts", body),
    onSuccess: (m) => {
      qc.invalidateQueries({ queryKey: ["manuscripts"] });
      qc.invalidateQueries({ queryKey: ["manuscript", m.id] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      toast.success(initial ? "Manuscript updated" : "Manuscript created. Now add its pages.");
      onSaved(m);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  async function submitWithFiles(c: number | null) {
    setProgress(0);
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    fd.append("title", form.title.trim());
    fd.append("language", form.language);
    if (c) fd.append("century", String(c));
    fd.append("material", form.material);
    if (form.site_id) fd.append("site_id", form.site_id);
    if (form.repository.trim()) fd.append("repository", form.repository.trim());
    if (form.description.trim()) fd.append("description", form.description.trim());
    if (form.license.trim()) fd.append("license", form.license.trim());
    if (form.source_url.trim()) fd.append("source_url", form.source_url.trim());
    fd.append("auto_ocr", "true");
    fd.append("engine", "auto");

    try {
      const res = await new Promise<{ manuscript: Manuscript; pages: unknown[] }>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API}/manuscripts/upload`);
        const s = loadSession();
        if (s) xhr.setRequestHeader("Authorization", `Bearer ${s.access_token}`);
        xhr.upload.onprogress = (e) => e.lengthComputable && setProgress((e.loaded / e.total) * 100);
        xhr.onload = () => {
          let b: any = null;
          try {
            b = JSON.parse(xhr.responseText);
          } catch {
            /* non-JSON */
          }
          if (xhr.status >= 200 && xhr.status < 300) resolve(b);
          else reject(new Error(b?.error?.message ?? `Upload failed (${xhr.status})`));
        };
        xhr.onerror = () => reject(new Error("Network error during upload"));
        xhr.send(fd);
      });

      qc.invalidateQueries({ queryKey: ["manuscripts"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      toast.success(`Manuscript created with ${files.length} pages — OCR reading started`);
      onSaved(res.manuscript);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setProgress(null);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const errs: typeof errors = {};
    if (form.title.trim().length < 2) errs.title = "Give the manuscript a title";
    const c = form.century ? Number(form.century) : null;
    if (form.century && (!Number.isInteger(c) || c! < 1 || c! > 21)) errs.century = "Century must be between 1 and 21";
    if (form.source_url && !/^https?:\/\//.test(form.source_url)) errs.source_url = "Start with http:// or https://";
    setErrors(errs);
    if (Object.keys(errs).length) return;

    if (!initial && files.length > 0) {
      submitWithFiles(c);
      return;
    }

    save.mutate({
      title: form.title.trim(),
      language: form.language,
      century: c,
      material: form.material,
      repository: form.repository.trim(),
      site_id: form.site_id || null,
      description: form.description.trim(),
      license: form.license.trim(),
      source_url: form.source_url.trim(),
    });
  }

  const set = (k: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) =>
    setForm({ ...form, [k]: e.target.value });

  return (
    <Modal
      open={open}
      onClose={() => progress === null && onClose()}
      title={initial ? "Edit manuscript" : "New manuscript"}
      subtitle="Catalogue details help people find and trust this text."
      width="max-w-2xl"
      footer={
        <>
          <Button onClick={onClose} disabled={progress !== null}>
            Cancel
          </Button>
          <Button variant="primary" type="submit" form="ms-form" loading={save.isPending || progress !== null}>
            {initial ? "Save changes" : files.length > 0 ? `Create & Upload ${files.length} Pages` : "Create manuscript"}
          </Button>
        </>
      }
    >
      <form id="ms-form" onSubmit={submit} className="grid gap-4 sm:grid-cols-2" noValidate>
        {/* Optional file upload area when creating a new manuscript */}
        {!initial && (
          <div className="sm:col-span-2 space-y-2">
            <span className="block text-[12.5px] font-medium text-ink-2">
              Manuscript Images or PDF (Optional)
            </span>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                addFiles(e.dataTransfer.files);
              }}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              className={clsx(
                "flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed p-4 text-center transition-colors",
                drag ? "border-accent bg-accent-soft/50" : "border-line-strong bg-surface-2/40 hover:bg-surface-2",
              )}
            >
              <ImageUp className="size-5 text-accent mb-1" />
              <div className="text-[13px] font-medium text-ink">
                {files.length > 0 ? "Add more folio images" : "Drop manuscript image files here to upload now"}
              </div>
              <div className="text-[11.5px] text-muted">JPG, PNG, WebP, TIFF or PDF · up to {MAX_MB} MB each</div>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={ACCEPT}
                className="hidden"
                onChange={(e) => {
                  addFiles(e.target.files);
                  e.target.value = "";
                }}
              />
            </div>

            {files.length > 0 && (
              <div className="max-h-32 overflow-y-auto rounded-lg border border-line divide-y divide-line bg-surface">
                {files.map((f, i) => (
                  <div key={`${f.name}-${i}`} className="flex items-center gap-2 p-1.5 text-[12.5px]">
                    {previews[i] ? (
                      <img src={previews[i]} alt="" className="size-7 rounded object-cover border border-line shrink-0" />
                    ) : f.type === "application/pdf" ? (
                      <FileText className="size-4 text-muted shrink-0" />
                    ) : (
                      <FileImage className="size-4 text-muted shrink-0" />
                    )}
                    <span className="flex-1 truncate text-ink">{f.name}</span>
                    <span className="text-[11px] text-muted tabular-nums">{(f.size / (1024 * 1024)).toFixed(1)} MB</span>
                    <button
                      type="button"
                      className="p-1 text-muted hover:text-ink"
                      onClick={() => removeFile(i)}
                      disabled={progress !== null}
                    >
                      <X className="size-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {progress !== null && (
              <div className="space-y-1">
                <Progress value={progress} />
                <div className="text-[11.5px] text-muted">
                  {progress < 100 ? `Uploading... ${progress.toFixed(0)}%` : "Creating manuscript & starting OCR..."}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="sm:col-span-2">
          <Field label="Title" error={errors.title}>
            <Input value={form.title} onChange={set("title")} placeholder="e.g. Bhagavad Gītā with commentary" />
          </Field>
        </div>
        <Field label="Language" hint="The script is chosen from the language">
          <Select value={form.language} onChange={set("language")}>
            {meta.data?.languages.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label} ({l.script})
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Century (CE)" error={errors.century} hint="Leave empty if undated">
          <Input type="number" min={1} max={21} value={form.century} onChange={set("century")} placeholder="18" />
        </Field>
        <Field label="Material">
          <Select value={form.material} onChange={set("material")}>
            {Object.entries(MATERIALS).map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Associated place" hint="Shown on the heritage map">
          <Select value={form.site_id} onChange={set("site_id")}>
            <option value="">None</option>
            {sites.data?.items.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
                {s.city ? ` — ${s.city}` : ""}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Repository / owner">
          <Input value={form.repository} onChange={set("repository")} placeholder="e.g. Oriental Institute, Vadodara" />
        </Field>
        <Field label="Licence">
          <Input value={form.license} onChange={set("license")} placeholder="e.g. CC BY-SA 4.0" />
        </Field>
        <div className="sm:col-span-2">
          <Field label="Source URL" error={errors.source_url}>
            <Input value={form.source_url} onChange={set("source_url")} placeholder="https://" />
          </Field>
        </div>
        <div className="sm:col-span-2">
          <Field label="Description">
            <Textarea rows={3} value={form.description} onChange={set("description")} placeholder="Contents, scribe, condition, provenance…" />
          </Field>
        </div>
      </form>
    </Modal>
  );
}
