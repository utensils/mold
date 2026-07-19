#!/usr/bin/env ruby
# frozen_string_literal: true

require "base64"
require "json"
require "net/http"
require "openssl"
require "uri"

API_BASE = ENV.fetch("ASC_API_BASE", "https://api.appstoreconnect.apple.com")

def b64(value)
  Base64.urlsafe_encode64(value, padding: false)
end

def token
  now = Time.now.to_i
  header = b64({ alg: "ES256", kid: ENV.fetch("ASC_KEY_ID"), typ: "JWT" }.to_json)
  payload = b64({ iss: ENV.fetch("ASC_ISSUER_ID"), iat: now, exp: now + 1_100, aud: "appstoreconnect-v1" }.to_json)
  signing_input = "#{header}.#{payload}"
  digest = OpenSSL::Digest::SHA256.digest(signing_input)
  signature = OpenSSL::PKey::EC.new(ENV.fetch("PRIVATE_KEY")).dsa_sign_asn1(digest)
  r, s = OpenSSL::ASN1.decode(signature).value.map(&:value)
  raw = [r.to_s(16).rjust(64, "0"), s.to_s(16).rjust(64, "0")].pack("H*H*")
  "#{signing_input}.#{b64(raw)}"
end

def request(method, path, body: nil)
  uri = URI("#{API_BASE}#{path}")
  request = method.new(uri)
  request["Authorization"] = "Bearer #{token}"
  if body
    request["Content-Type"] = "application/json"
    request.body = JSON.generate(body)
  end
  Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https", open_timeout: 15, read_timeout: 60) do |http|
    http.request(request)
  end
end

def json_request(method, path, body: nil, allow_conflict: false)
  response = request(method, path, body: body)
  return JSON.parse(response.body) if response.is_a?(Net::HTTPSuccess) && !response.body.empty?
  return {} if response.is_a?(Net::HTTPSuccess)
  return {} if allow_conflict && response.code.to_i == 409

  raise "App Store Connect #{response.code}: #{response.body}"
end

def query(path, params)
  "#{path}?#{URI.encode_www_form(params)}"
end

bundle_id = ENV.fetch("APP_BUNDLE_ID")
email = ENV.fetch("TESTFLIGHT_TESTER_EMAIL")
group_name = ENV.fetch("TESTFLIGHT_GROUP_NAME", "Mold Internal")

apps = json_request(
  Net::HTTP::Get,
  query("/v1/apps", { "filter[bundleId]" => bundle_id, "fields[apps]" => "name,bundleId", "limit" => "1" })
).fetch("data")
app = apps.first or abort "No App Store Connect app for #{bundle_id}"

groups = json_request(
  Net::HTTP::Get,
  query(
    "/v1/betaGroups",
    { "filter[app]" => app.fetch("id"), "fields[betaGroups]" => "name,isInternalGroup,hasAccessToAllBuilds", "limit" => "200" }
  )
).fetch("data")
group = groups.find do |candidate|
  candidate.dig("attributes", "name") == group_name && candidate.dig("attributes", "isInternalGroup")
end

unless group
  group = json_request(
    Net::HTTP::Post,
    "/v1/betaGroups",
    body: {
      data: {
        type: "betaGroups",
        attributes: {
          name: group_name,
          isInternalGroup: true,
          hasAccessToAllBuilds: true,
          feedbackEnabled: true
        },
        relationships: {
          app: { data: { type: "apps", id: app.fetch("id") } }
        }
      }
    }
  ).fetch("data")
  puts "Created internal TestFlight group #{group_name}."
else
  group = json_request(
    Net::HTTP::Patch,
    "/v1/betaGroups/#{group.fetch('id')}",
    body: {
      data: {
        type: "betaGroups",
        id: group.fetch("id"),
        attributes: { hasAccessToAllBuilds: true, feedbackEnabled: true }
      }
    }
  ).fetch("data")
end

testers = json_request(
  Net::HTTP::Get,
  query("/v1/betaTesters", { "filter[email]" => email, "fields[betaTesters]" => "email,state", "limit" => "20" })
).fetch("data")
tester = testers.find { |candidate| candidate.dig("attributes", "email")&.casecmp?(email) }

if tester
  json_request(
    Net::HTTP::Post,
    "/v1/betaGroups/#{group.fetch('id')}/relationships/betaTesters",
    body: { data: [{ type: "betaTesters", id: tester.fetch("id") }] },
    allow_conflict: true
  )
else
  tester = json_request(
    Net::HTTP::Post,
    "/v1/betaTesters",
    body: {
      data: {
        type: "betaTesters",
        attributes: { email: email, firstName: "James", lastName: "Brink" },
        relationships: {
          betaGroups: { data: [{ type: "betaGroups", id: group.fetch("id") }] }
        }
      }
    }
  ).fetch("data")
end

members = json_request(
  Net::HTTP::Get,
  query(
    "/v1/betaGroups/#{group.fetch('id')}/betaTesters",
    { "fields[betaTesters]" => "email,state", "limit" => "200" }
  )
).fetch("data")
member = members.find { |candidate| candidate.dig("attributes", "email")&.casecmp?(email) }
abort "TestFlight tester #{email} was not present in #{group_name} after assignment" unless member

state = member.dig("attributes", "state") || tester.dig("attributes", "state") || "assigned"
puts "Verified TestFlight access for #{email} in #{group_name} (#{state})."
