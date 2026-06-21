export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
      <span className="font-medium">Error:</span> {message}
    </div>
  );
}
