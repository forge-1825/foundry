import React from 'react';

export function Alert({ className, variant = 'default', children, ...props }) {
  const variantClasses = {
    default: 'bg-background text-foreground',
    destructive: 'bg-destructive/15 text-destructive border-destructive/50',
    success: 'bg-success/15 text-success border-success/50',
    warning: 'bg-warning/15 text-warning border-warning/50',
    info: 'bg-info/15 text-info border-info/50',
  };

  return (
    <div
      role="alert"
      className={`relative w-full rounded-lg border p-4 [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg+div]:translate-y-[-3px] [&:has(svg)]:pl-11 ${variantClasses[variant]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function AlertTitle({ className, children, ...props }) {
  return (
    <h5
      className={`mb-1 font-medium leading-none tracking-tight ${className}`}
      {...props}
    >
      {children}
    </h5>
  );
}

export function AlertDescription({ className, children, ...props }) {
  return (
    <div
      className={`text-sm [&_p]:leading-relaxed ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
