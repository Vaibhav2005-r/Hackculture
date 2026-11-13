import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { ChevronDown, ChevronUp, FileImage, FileText, ImageUp, Sparkles, X } from "lucide-react";
import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { API, loadSession } from "../lib/api";
import { MATERIALS } from "../lib/format";
import { useMeta, useSites } from "../lib/queries";
import type { Manuscript, Page } from "../lib/types";
import { Button, Field, Input, Modal, Progress, Select, Textarea } from "./ui";

const ACCEPT = "image/jpeg,image/png,image/tiff,image/webp,image/bmp,application/pdf";
const MAX_MB = 25;

export function fileToTitle(name: string): string {
  const base = name.replace(/\.[^/.]+$/, "");
  const cleaned = base.replace(/[_-]+/g, " ").trim();
  return cleaned
    .split(/\s+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

interface UploadResponse {
  manuscript: Manuscript;
  pages: Page[];
  jobs: unknown[];
}

function directUpload(form: FormData, onProgress: (p: number) => void): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/manuscripts/upload`);
    const s = loadSession();
    if (s) xhr.setRequestHeader("Authorization", `Bearer ${s.access_token}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress((e.loaded / e.total) * 100);
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body?.error?.message ?? `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(form);
  });
}

export function DirectUploadModal({
  open,
  onClose,
  initialFiles,
}: {
  open: boolean;
  onClose: () => void;
  initialFiles?: File[];
}) {
  const meta = useMeta();
  const sites = useSites();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const formId = useId();

  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);

  // Core metadata
  const [title, setTitle] = useState("");
  const [titleTouched, setTitleTouched] = useState(false);
  const [language, setLanguage] = useState("sa");
  const [autoOcr, setAutoOcr] = useState(true);
  const [engine, setEngine] = useState("auto");

  // Optional catalogue metadata
  const [showDetails, setShowDetails] = useState(false);
  const [century, setCentury] = useState("");
  const [material, setMaterial] = useState("paper");
  const [siteId, setSiteId] = useState("");
  const [repository, setRepository] = useState("");
  const [description, setDescription] = useState("");
  const [license, setLicense] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");

  const [errors, setErrors] = useState<{ title?: string; century?: string; source_url?: string }>({});

  useEffect(() => {
    if (!open) {
      setFiles([]);
      previews.forEach((u) => URL.revokeObjectURL(u));
      setPreviews([]);
      setTitle("");
      setTitleTouched(false);
      setProgress(null);
      setErrors({});
      setShowDetails(false);
      return;
    }
    if (initialFiles && initialFiles.length > 0) {
      addFiles(initialFiles);
    }
  }, [open, initialFiles]);

  function addFiles(newFiles: File[] | FileList | null) {
    if (!newFiles) return;
    const ok: File[] = [];
    for (const f of Array.from(newFiles)) {
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
      if (!titleTouched && merged.length > 0) {
        setTitle(fileToTitle(merged[0].name));
      }
      return merged;
    });

    const urls = ok.map((f) => (f.type.startsWith("image/") ? URL.createObjectURL(f) : ""));
    setPreviews((prev) => [...prev, ...urls].slice(0, 50));
  }

  function removeFile(index: number) {
    if (previews[index]) URL.revokeObjectURL(previews[index]);
    setFiles((prev) => {
      const next = prev.filter((_, i) => i !== index);
      if (!titleTouched && next.length > 0) {
        setTitle(fileToTitle(next[0].name));
      } else if (next.length === 0 && !titleTouched) {
        setTitle("");
      }
      return next;
    });
    setPreviews((prev) => prev.filter((_, i) => i !== index));
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (files.length === 0) {
      toast.error("Please select at least one manuscript image or PDF");
      return;
    }

    const errs: typeof errors = {};
    if (title.trim().length < 2) errs.title = "Give the manuscript a title";
    const c = century ? Number(century) : null;
    if (century && (!Number.isInteger(c) || c! < 1 || c! > 21)) errs.century = "Century must be between 1 and 21";
    if (sourceUrl && !/^https?:\/\//.test(sourceUrl)) errs.source_url = "Start with http:// or https://";
    setErrors(errs);
    if (Object.keys(errs).length) return;

    setProgress(0);
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    fd.append("title", title.trim());
    fd.append("language", language);
    if (c) fd.append("century", String(c));
    fd.append("material", material);
    if (siteId) fd.append("site_id", siteId);
    if (repository.trim()) fd.append("repository", repository.trim());
    if (description.trim()) fd.append("description", description.trim());
    if (license.trim()) fd.append("license", license.trim());
    if (sourceUrl.trim()) fd.append("source_url", sourceUrl.trim());
    fd.append("auto_ocr", String(autoOcr));
    fd.append("engine", engine);

    try {
      const res = await directUpload(fd, setProgress);
      qc.invalidateQueries({ queryKey: ["manuscripts"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      toast.success(
        `"${res.manuscript.title}" uploaded (${res.pages.length} ${res.pages.length === 1 ? "page" : "pages"})${autoOcr ? " — OCR in progress" : ""}`,
      );
      onClose();
      navigate(`/manuscripts/${res.manuscript.id}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setProgress(null);
    }
  }

  return (
    <Modal
      open={open}
      onClose={() => progress === null && onClose()}
      title="Upload Manuscript Images"
      subtitle="Directly upload photos or scans of manuscript folios to digitise and read them."
      width="max-w-2xl"
      footer={
        <>
          <Button onClick={onClose} disabled={progress !== null}>
            Cancel
          </Button>
          <Button
            variant="primary"
            type="submit"
            form={formId}
            loading={progress !== null}
            disabled={files.length === 0}
            icon={<Sparkles className="size-4" />}
          >
            {files.length > 0 ? `Upload ${files.length} ${files.length === 1 ? "page" : "pages"} & Read` : "Select images first"}
          </Button>
        </>
      }
    >
      <form id={formId} onSubmit={submit} className="space-y-4" noValidate>
        {/* Drop zone */}
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
            "flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition-all",
            drag ? "border-accent bg-accent-soft/50 scale-[1.01]" : "border-line-strong bg-surface-2/50 hover:border-accent/60 hover:bg-surface-2",
          )}
        >
          <div className="flex size-12 items-center justify-center rounded-2xl bg-surface text-accent shadow-card">
            <ImageUp className="size-6" />
          </div>
          <div className="mt-2.5 text-[14px] font-semibold text-ink">
            Drop manuscript image folios or PDF here
          </div>
          <div className="mt-0.5 text-[12px] text-muted">
            JPG, PNG, WebP, TIFF, BMP or PDF · up to {MAX_MB} MB per file · multiple files supported
          </div>
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

        {/* Selected files preview */}
        {files.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[12.5px] font-medium text-ink-2">
              <span>{files.length} {files.length === 1 ? "folio" : "folios"} selected:</span>
              <button
                type="button"
                className="text-muted hover:text-bad text-[12px]"
                onClick={() => {
                  previews.forEach((u) => URL.revokeObjectURL(u));
                  setPreviews([]);
                  setFiles([]);
                  if (!titleTouched) setTitle("");
                }}
                disabled={progress !== null}
              >
                Clear all
              </button>
            </div>
            <div className="max-h-48 overflow-y-auto rounded-xl border border-line divide-y divide-line bg-surface">
              {files.map((f, i) => (
                <div key={`${f.name}-${i}`} className="flex items-center gap-3 p-2 text-[13px]">
                  <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-surface-2 text-[11px] font-semibold text-muted">
                    {i + 1}
                  </span>
                  {previews[i] ? (
                    <img
                      src={previews[i]}
                      alt=""
                      className="size-9 rounded-md object-cover border border-line shrink-0"
                    />
                  ) : f.type === "application/pdf" ? (
                    <div className="flex size-9 items-center justify-center rounded-md bg-surface-2 text-muted shrink-0">
                      <FileText className="size-4" />
                    </div>
                  ) : (
                    <div className="flex size-9 items-center justify-center rounded-md bg-surface-2 text-muted shrink-0">
                      <FileImage className="size-4" />
                    </div>
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-medium text-ink">{f.name}</div>
                    <div className="text-[11px] text-muted">{(f.size / (1024 * 1024)).toFixed(2)} MB</div>
                  </div>
                  <button
                    type="button"
                    className="rounded p-1 text-muted hover:bg-surface-2 hover:text-ink shrink-0"
                    onClick={() => removeFile(i)}
                    aria-label={`Remove ${f.name}`}
                    disabled={progress !== null}
                  >
                    <X className="size-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Title and Language */}
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <Field label="Manuscript Title" error={errors.title} hint="Auto-derived from image name, or customize here">
              <Input
                value={title}
                onChange={(e) => {
                  setTitle(e.target.value);
                  setTitleTouched(true);
                }}
                placeholder="e.g. Advaita Viveka Folio 1"
              />
            </Field>
          </div>
          <Field label="Language & Script" hint="Recognizer uses the script of the chosen language">
            <Select value={language} onChange={(e) => setLanguage(e.target.value)}>
              {meta.data?.languages.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label} ({l.script})
                </option>
              ))}
            </Select>
          </Field>
          <Field label="OCR Processing">
            <div className="space-y-1.5 pt-1">
              <label className="flex items-center gap-2 text-[13px] text-ink cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoOcr}
                  onChange={(e) => setAutoOcr(e.target.checked)}
                  className="size-4 rounded border-line text-accent focus:ring-accent"
                />
                Read and extract text right away
              </label>
              {autoOcr && (
                <Select
                  value={engine}
                  onChange={(e) => setEngine(e.target.value)}
                  className="h-8 text-[12px]"
                >
                  <option value="auto">Engine: Automatic (recommended)</option>
                  {meta.data?.engines.filter((eng) => eng.available).map((eng) => (
                    <option key={eng.name} value={eng.name}>
                      {eng.label}
                    </option>
                  ))}
                </Select>
              )}
            </div>
          </Field>
        </div>

        {/* Optional Collapsible Details */}
        <div className="border-t border-line pt-3">
          <button
            type="button"
            className="flex items-center gap-1.5 text-[12.5px] font-medium text-muted hover:text-ink transition-colors"
            onClick={() => setShowDetails((s) => !s)}
          >
            {showDetails ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
            {showDetails ? "Hide additional catalogue fields" : "Add catalogue details (century, repository, heritage site, etc.)"}
          </button>

          {showDetails && (
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <Field label="Century (CE)" error={errors.century} hint="e.g. 17 or 18 (leave blank if unknown)">
                <Input
                  type="number"
                  min={1}
                  max={21}
                  value={century}
                  onChange={(e) => setCentury(e.target.value)}
                  placeholder="18"
                />
              </Field>
              <Field label="Material">
                <Select value={material} onChange={(e) => setMaterial(e.target.value)}>
                  {Object.entries(MATERIALS).map(([v, l]) => (
                    <option key={v} value={v}>
                      {l}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Associated Heritage Place" hint="Links manuscript to the heritage map">
                <Select value={siteId} onChange={(e) => setSiteId(e.target.value)}>
                  <option value="">None</option>
                  {sites.data?.items.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                      {s.city ? ` — ${s.city}` : ""}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Repository / Archive">
                <Input
                  value={repository}
                  onChange={(e) => setRepository(e.target.value)}
                  placeholder="e.g. Bhandarkar Oriental Research Institute"
                />
              </Field>
              <Field label="Licence">
                <Input
                  value={license}
                  onChange={(e) => setLicense(e.target.value)}
                  placeholder="e.g. CC0, CC BY 4.0, Public Domain"
                />
              </Field>
              <Field label="Source URL" error={errors.source_url}>
                <Input
                  value={sourceUrl}
                  onChange={(e) => setSourceUrl(e.target.value)}
                  placeholder="https://..."
                />
              </Field>
              <div className="sm:col-span-2">
                <Field label="Description / Provenance">
                  <Textarea
                    rows={2}
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Subject matter, scribe notes, condition..."
                  />
                </Field>
              </div>
            </div>
          )}
        </div>

        {/* Progress indicator */}
        {progress !== null && (
          <div className="space-y-1.5 rounded-xl bg-surface-2 p-3">
            <div className="flex justify-between text-[12px] font-medium text-ink">
              <span>Uploading manuscript images...</span>
              <span className="tabular-nums">{progress.toFixed(0)}%</span>
            </div>
            <Progress value={progress} />
            <div className="text-[11px] text-muted">
              {progress < 100
                ? "Sending files to server..."
                : "Encoding pages, saving to storage and starting OCR pipeline..."}
            </div>
          </div>
        )}
      </form>
    </Modal>
  );
}

