import React, { createContext, useContext, useState } from 'react';

const SelectContext = createContext(null);

export function Select({ children, value, onValueChange, ...props }) {
  const [isOpen, setIsOpen] = useState(false);
  
  const handleSelect = (newValue) => {
    onValueChange(newValue);
    setIsOpen(false);
  };
  
  return (
    <SelectContext.Provider value={{ value, onValueChange: handleSelect, isOpen, setIsOpen }}>
      <div className="relative" {...props}>
        {children}
      </div>
    </SelectContext.Provider>
  );
}

export function SelectTrigger({ className, children, ...props }) {
  const { isOpen, setIsOpen } = useContext(SelectContext);
  
  return (
    <button
      type="button"
      className={`flex items-center justify-between w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      onClick={() => setIsOpen(!isOpen)}
      {...props}
    >
      {children}
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={`ml-2 h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
      >
        <polyline points="6 9 12 15 18 9"></polyline>
      </svg>
    </button>
  );
}

export function SelectValue({ placeholder }) {
  const { value } = useContext(SelectContext);
  
  return (
    <span className="truncate">
      {value || placeholder}
    </span>
  );
}

export function SelectContent({ className, children, ...props }) {
  const { isOpen } = useContext(SelectContext);
  
  if (!isOpen) return null;
  
  return (
    <div
      className={`absolute z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md animate-in fade-in-80 mt-1 w-full ${className}`}
      {...props}
    >
      <div className="p-1">
        {children}
      </div>
    </div>
  );
}

export function SelectItem({ className, children, value, ...props }) {
  const { value: selectedValue, onValueChange } = useContext(SelectContext);
  const isSelected = selectedValue === value;
  
  return (
    <div
      className={`relative flex cursor-default select-none items-center rounded-sm py-1.5 px-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 ${isSelected ? 'bg-accent text-accent-foreground' : ''} ${className}`}
      onClick={() => onValueChange(value)}
      {...props}
    >
      <span className="flex-1">{children}</span>
      {isSelected && (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-4 w-4"
        >
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      )}
    </div>
  );
}
