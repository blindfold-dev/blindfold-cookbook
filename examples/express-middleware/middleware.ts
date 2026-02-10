/**
 * Express middleware that automatically tokenizes PII in request bodies
 * and provides a detokenize helper for responses.
 */

import { Blindfold, type TokenizeResponse } from "@blindfold/sdk";
import type { Request, Response, NextFunction } from "express";

interface BlindfoldRequest extends Request {
  blindfold?: {
    originalText: string;
    tokenizedText: string;
    mapping: Record<string, string>;
    detokenize: (text: string) => string;
  };
}

interface MiddlewareOptions {
  apiKey: string;
  policy?: string;
  textField?: string;
}

export function blindfoldMiddleware(options: MiddlewareOptions) {
  const blindfold = new Blindfold({ apiKey: options.apiKey });
  const policy = options.policy ?? "basic";
  const textField = options.textField ?? "text";

  return async (req: BlindfoldRequest, res: Response, next: NextFunction) => {
    const text = req.body?.[textField];
    if (!text || typeof text !== "string") {
      return next();
    }

    try {
      const result: TokenizeResponse = await blindfold.tokenize(text, {
        policy,
      });

      // Replace request body text with tokenized version
      req.body[textField] = result.text;

      // Attach Blindfold context for downstream handlers
      req.blindfold = {
        originalText: text,
        tokenizedText: result.text,
        mapping: result.mapping,
        detokenize: (responseText: string) =>
          blindfold.detokenize(responseText, result.mapping).text,
      };

      next();
    } catch (err) {
      next(err);
    }
  };
}

export type { BlindfoldRequest };
