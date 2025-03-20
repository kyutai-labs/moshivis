import { useCallback, useState } from "react";
import { z } from "zod";

const validateEmail = z.string().email();

export const useUserEmail = (isBypass: boolean, init_value: string) => {
  const [userEmail, setUserEmail] = useState<string>(init_value);
  const [error, setError] = useState<string | null>(null);

  const validate = useCallback((email: string) => {
    if (isBypass) {
      setError(null);
      return true;
    }
    const result = validateEmail.safeParse(email);
    if (result.success) {
      setError(null);
      sessionStorage.setItem("userEmail", email.toString());
      return true;
    }
    setError('Invalid email address');
    return false;
  }, [setError]);
  return { userEmail, setUserEmail, error, validate };
}
