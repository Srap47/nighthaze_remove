import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud } from 'lucide-react';

const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB — matches backend max_image_size_mb

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export function DropZone({ onFileSelect, disabled = false }: DropZoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect],
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } =
    useDropzone({
      onDrop,
      accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] },
      maxFiles: 1,
      maxSize: MAX_SIZE_BYTES,
      multiple: false,
      disabled,
    });

  // Surface the first rejection reason inline.
  const rejectionError = fileRejections[0]?.errors[0];
  let rejectionMessage: string | null = null;
  if (rejectionError) {
    switch (rejectionError.code) {
      case 'file-too-large':
        rejectionMessage = 'That image is larger than 10MB. Please choose a smaller file.';
        break;
      case 'file-invalid-type':
        rejectionMessage = 'Unsupported file type. Please use JPEG, PNG, or WebP.';
        break;
      case 'too-many-files':
        rejectionMessage = 'Please drop only one image at a time.';
        break;
      default:
        rejectionMessage = rejectionError.message;
    }
  }

  const stateClasses = isDragActive
    ? 'border-primary bg-primary/5'
    : 'border-white/20 hover:border-white/30 hover:bg-white/[0.02]';

  return (
    <div>
      <div
        {...getRootProps({
          className: `flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 text-center transition ${stateClasses} ${
            disabled ? 'cursor-not-allowed opacity-50' : ''
          }`,
        })}
      >
        <input {...getInputProps()} />
        <UploadCloud className="h-10 w-10 text-white/40" />
        <h3 className="mt-4 text-base font-semibold">
          Drag &amp; drop a nighttime image
        </h3>
        <p className="mt-1 text-sm text-white/50">
          or click to browse — JPEG, PNG, WebP up to 10MB
        </p>
      </div>

      {rejectionMessage && (
        <p className="mt-3 text-center text-sm text-red-400">{rejectionMessage}</p>
      )}
    </div>
  );
}
